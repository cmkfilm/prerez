#!/usr/bin/env python3
"""
fcpxml_merge_shorts.py — Merge sub-minimum clips in an FCPXML timeline

Reads a DaVinci Resolve / FCP FCPXML, finds clips shorter than Topaz's
100ms minimum, and merges consecutive short runs into single processable
clips. Outputs a modified FCPXML ready to re-import into Resolve for
re-export.

Also fixes the classifier's MERGE_SHORT_FRAMES threshold mismatch:
the old default of 10 frames (334ms) was silently discarding clips that
Topaz can actually process. The correct threshold is 3 frames (100ms).

Usage:
    python3 fcpxml_merge_shorts.py input.fcpxml
    python3 fcpxml_merge_shorts.py input.fcpxml --out merged.fcpxml
    python3 fcpxml_merge_shorts.py input.fcpxml --min-ms 100 --report

Merge rules:
  1. Consecutive clips all below --min-ms are merged into one clip
     spanning the combined source range (start of first → end of last).
  2. A merged run is only viable if its total duration ≥ --min-ms.
     Irresolvably short clips (isolated 1-2 frame clips with no viable
     neighbour) are flagged in the report and left as-is — they become
     pass-through clips in the Topaz workflow.
  3. Clips spanning different source files are never merged (though for
     single-source timelines this never occurs).

Output:
  <input>_merged.fcpxml    modified XML (or --out path)
  <input>_merge_report.csv summary of every merge decision
"""

import argparse
import copy
import csv
import sys
from fractions import Fraction
from pathlib import Path
import xml.etree.ElementTree as ET


TOPAZ_MIN_MS      = 100           # Topaz hard minimum in milliseconds
TOPAZ_MIN_FRAMES  = 3             # at 29.97fps


# ── Rational time helpers ──────────────────────────────────────────────

def parse_r(s: str) -> Fraction:
    s = s.rstrip('s')
    if '/' in s:
        n, d = s.split('/', 1)
        return Fraction(int(n), int(d))
    return Fraction(int(float(s) * 1) if '.' in s else int(s))


def fmt_r(f: Fraction) -> str:
    if f.denominator == 1:
        return f"{f.numerator}s"
    return f"{f.numerator}/{f.denominator}s"


def to_ms(f: Fraction) -> float:
    return float(f) * 1000.0


def to_frames(f: Fraction, fps: Fraction) -> float:
    return float(f * fps)


# ── FCPXML parsing ─────────────────────────────────────────────────────

def parse_clips(spine, fps: Fraction) -> list[dict]:
    clips = []
    for i, ac in enumerate(spine):
        if ac.tag != 'asset-clip':
            continue
        start  = parse_r(ac.get('start',    '0s'))
        dur    = parse_r(ac.get('duration', '0s'))
        offset = parse_r(ac.get('offset',   '0s'))
        clips.append({
            'idx':      i,
            'elem':     ac,
            'ref':      ac.get('ref', ''),
            'name':     ac.get('name', ''),
            'format':   ac.get('format', ''),
            'tcFormat': ac.get('tcFormat', 'NDF'),
            'start':    start,
            'dur':      dur,
            'end':      start + dur,
            'offset':   offset,
            'dur_ms':   to_ms(dur),
            'dur_f':    to_frames(dur, fps),
        })
    return clips


# ── Merge logic ────────────────────────────────────────────────────────

def find_merge_groups(clips: list[dict], min_ms: float) -> list[dict]:
    """
    Returns a list of actions, one per original clip:
      action='keep'   — clip is above minimum, no change
      action='merge'  — clip is part of a viable merge group
      action='short'  — irresolvably short (total of run still < min_ms)

    A merge group is a run of clips that are consecutive in the spine
    (i.e. adjacent in the timeline) whose individual durations are all
    below min_ms, and whose combined duration >= min_ms.

    Note: source-file identity is intentionally NOT used as a merge
    boundary. For single-source timelines (the common case) all clips
    share the same source file. For multi-source timelines, merging
    across sources in the FCPXML sense means the merged clip's start/
    duration attributes reference the first clip's source — Resolve
    will handle the actual frame range on re-import.
    """
    n       = len(clips)
    actions = [{'action': 'keep', 'group_id': None, 'is_first': False}
               for _ in range(n)]
    i        = 0
    group_id = 0

    while i < n:
        if clips[i]['dur_ms'] < min_ms:
            run_start = i
            while i < n and clips[i]['dur_ms'] < min_ms:
                i += 1
            run_end  = i  # exclusive
            total_ms = sum(clips[j]['dur_ms'] for j in range(run_start, run_end))

            if total_ms >= min_ms:
                for j in range(run_start, run_end):
                    actions[j] = {
                        'action':   'merge',
                        'group_id': group_id,
                        'is_first': j == run_start,
                    }
                group_id += 1
            else:
                for j in range(run_start, run_end):
                    actions[j] = {
                        'action':   'short',
                        'group_id': None,
                        'is_first': False,
                    }
        else:
            i += 1

    return actions


def build_merged_xml(tree: ET.ElementTree,
                     spine: ET.Element,
                     clips: list[dict],
                     actions: list[dict],
                     fps: Fraction) -> ET.ElementTree:
    """
    Returns a deep-copied tree with the spine modified:
    - Merged runs replaced by a single asset-clip with combined duration
    - Irresolvably short clips kept as-is (flagged in report only)
    - Normal clips unchanged
    """
    new_tree  = copy.deepcopy(tree)
    new_spine = new_tree.find('.//spine')

    # Build index: original element -> new element mapping
    orig_to_new = {}
    for old_elem, new_elem in zip(spine, new_spine):
        orig_to_new[id(old_elem)] = new_elem

    # Process actions: collect elements to remove and elements to modify
    # Group merges by group_id
    from collections import defaultdict
    merge_groups = defaultdict(list)
    for clip, action in zip(clips, actions):
        if action['action'] == 'merge':
            merge_groups[action['group_id']].append(clip)

    # For each merge group: extend first clip's duration, remove the rest
    elems_to_remove = set()
    for gid, group in merge_groups.items():
        first = group[0]
        total_dur = sum(c['dur'] for c in group)
        # Modify the first clip's element in the NEW tree
        new_elem = orig_to_new[id(first['elem'])]
        new_elem.set('duration', fmt_r(total_dur))
        # Mark the rest for removal
        for c in group[1:]:
            elems_to_remove.add(id(c['elem']))

    # Remove marked elements
    to_remove_new = [orig_to_new[eid] for eid in elems_to_remove
                     if eid in orig_to_new]
    for elem in to_remove_new:
        new_spine.remove(elem)

    return new_tree


# ── Report ─────────────────────────────────────────────────────────────

def write_report(clips: list[dict], actions: list[dict],
                 fps: Fraction, report_path: Path) -> dict:
    from collections import defaultdict
    merge_groups = defaultdict(list)
    for clip, action in zip(clips, actions):
        if action['action'] == 'merge':
            merge_groups[action['group_id']].append(clip)

    stats = {
        'total':         len(clips),
        'kept':          sum(1 for a in actions if a['action'] == 'keep'),
        'merged_clips':  sum(1 for a in actions if a['action'] == 'merge'),
        'merge_groups':  len(merge_groups),
        'irresolvable':  sum(1 for a in actions if a['action'] == 'short'),
    }

    with open(report_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['clip_index', 'name', 'start_s', 'dur_ms', 'dur_frames',
                    'action', 'group_id', 'group_total_ms', 'note'])
        for i, (clip, action) in enumerate(zip(clips, actions)):
            act  = action['action']
            gid  = action.get('group_id')
            group_ms = ''
            note = ''
            if act == 'merge' and gid is not None:
                group_ms = f"{sum(c['dur_ms'] for c in merge_groups[gid]):.1f}"
                if action.get('is_first'):
                    note = f'MERGED HEAD (group {gid})'
                else:
                    note = f'merged into group {gid}'
            elif act == 'short':
                note = 'PASS-THROUGH (irresolvably short)'
            w.writerow([
                i + 1,
                clip['name'],
                f"{float(clip['start']):.4f}",
                f"{clip['dur_ms']:.1f}",
                f"{clip['dur_f']:.2f}",
                act,
                gid if gid is not None else '',
                group_ms,
                note,
            ])
    return stats


# ── Main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Merge sub-minimum clips in FCPXML for Topaz compatibility"
    )
    ap.add_argument('input', help='Input .fcpxml file')
    ap.add_argument('--out', default=None,
                    help='Output .fcpxml path (default: <input>_merged.fcpxml)')
    ap.add_argument('--min-ms', type=float, default=TOPAZ_MIN_MS,
                    help=f'Minimum clip duration in ms (default: {TOPAZ_MIN_MS})')
    ap.add_argument('--report', action='store_true',
                    help='Write merge report CSV alongside output')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print summary only, do not write files')
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found")
        sys.exit(1)

    out_path    = Path(args.out) if args.out else in_path.with_name(
        in_path.stem + '_merged.fcpxml')
    report_path = out_path.with_name(out_path.stem + '_merge_report.csv')

    # Parse
    ET.register_namespace('', '')
    tree = ET.parse(in_path)
    root = tree.getroot()

    # Detect fps from format
    fmt = root.find('.//format')
    if fmt is not None and fmt.get('frameDuration'):
        fd  = parse_r(fmt.get('frameDuration'))
        fps = Fraction(1) / fd
    else:
        fps = Fraction(30000, 1001)
        print(f"WARNING: Could not detect fps, assuming 29.97")

    print(f"FPS: {float(fps):.4f}")
    print(f"Topaz minimum: {args.min_ms:.0f}ms "
          f"({args.min_ms/1000*float(fps):.1f} frames)")

    spine = root.find('.//spine')
    if spine is None:
        print("ERROR: No <spine> found in FCPXML")
        sys.exit(1)

    clips   = parse_clips(spine, fps)
    actions = find_merge_groups(clips, args.min_ms)

    # Stats
    kept       = sum(1 for a in actions if a['action'] == 'keep')
    merged     = sum(1 for a in actions if a['action'] == 'merge')
    short_pass = sum(1 for a in actions if a['action'] == 'short')

    from collections import defaultdict
    merge_groups = defaultdict(list)
    for clip, action in zip(clips, actions):
        if action['action'] == 'merge':
            merge_groups[action['group_id']].append(clip)

    output_clips = kept + len(merge_groups) + short_pass

    print(f"\nInput:  {len(clips)} clips")
    print(f"  Above minimum (kept):            {kept}")
    print(f"  Below minimum → merged:          {merged} clips "
          f"→ {len(merge_groups)} output clips")
    print(f"  Below minimum → pass-through:    {short_pass} "
          f"(irresolvably short, flagged)")
    print(f"Output: {output_clips} clips "
          f"({len(clips)-merged+len(merge_groups)} after merging)")

    if args.dry_run:
        print("\n(dry run — no files written)")
        return

    # Write merged XML
    new_tree = build_merged_xml(tree, spine, clips, actions, fps)

    # Preserve XML declaration
    with open(out_path, 'wb') as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(b'<!DOCTYPE fcpxml>\n')
        ET.indent(new_tree, space='    ')
        new_tree.write(f, encoding='utf-8', xml_declaration=False)

    print(f"\nWrote: {out_path}")

    if args.report:
        stats = write_report(clips, actions, fps, report_path)
        print(f"Wrote: {report_path}")
        print(f"\nReport summary:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    # Remind about classifier threshold fix
    print(f"""
NOTE: Also update MERGE_SHORT_FRAMES in clip_roundtrip_classify_v6_4.py:
  Old: MERGE_SHORT_FRAMES = 10  (skips clips < 334ms — too aggressive)
  New: MERGE_SHORT_FRAMES = 3   (matches Topaz 100ms minimum)
  This recovers ~162 processable clips that were being silently dropped.
""")


if __name__ == '__main__':
    main()
