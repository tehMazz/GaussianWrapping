"""Hollowing: no geometry where none is needed (unseen, free or see-through space).

ledger.py      H1 visibility ledger, unseen pruning
free_space.py  H2 free-space prior + ray-carved free grid, H3 opacity entropy / decay
carve.py       H4 see-through carving
schedule.py    HollowSchedule: the train/finetune hook
"""
