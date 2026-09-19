# E4 annotation inputs and outputs. Three people label the same 70 emails
# blind, from one handout; the protocol is the docs/PREREGISTRATION.md section
# 9 entry dated 2026-09-19.
#
# annotator2_sheet.csv is the handout exactly as it was sent: built from dataset
# v1 and deliberately not regenerated, because a regenerated sheet would make
# the record describe a file the annotator never saw. One consequence is that it
# still carries the case-007 email address that golden_dataset.json replaced
# when it moved to v1.1 on 2026-09-17 -- an invented account holder at a domain
# that turned out to belong to a real company. The address is not reprinted
# here; why the residue is kept rather than scrubbed, and what moved instead, is
# the docs/PREREGISTRATION.md section 9 entry headed "one email address in
# golden_dataset.json belonged to a real company; the dataset moves to v1.1".
# annotator2_instructions_zh.md is the instruction sheet all three worked from.
#
# Returned files are committed byte for byte -- A1's as received, A2's and
# A3's as their notes-blanked copies, per the rule further down -- under names
# that carry the annotator id and the date and no person's name. Each one's
# sha256 was
# recorded in section 9 on the day it arrived and before it was read against
# anything, so the ordering is checkable by hashing the file at the import
# commit and matching it to the entry that precedes that commit:
#
#   a1_category_labels_2026-09-19.csv  774e62bc...02cfbb   70 category labels
#   a1_summary_review_2026-09-19.csv   48b1d1d3...6a7759   70 ok/edit verdicts
#
# A2's and A3's sheets arrive under two hashes each: the file as received, and
# the copy with the notes column emptied, which is the one committed here. The
# original stays with the author. Those two are outside the project and were
# promised anonymity and a published category answer, not a published notes
# column; blanking touches that one column, so the labels, the case ids and the
# email bodies are unchanged. blank_sheet_notes.py makes the copy and prints
# both hashes. A1's notes came back empty, so the step is a no-op on that sheet
# and its two hashes are the single value above.
#
# Derived, written by the importers and regenerable from the CSVs above:
#
#   annotations/<id>_labels.json    import_annotations.py --annotator A1|A2|A3
#   summary_review_a1.json          import_summary_review.py: counts and the
#                                   ids marked edit, which is all section 9
#                                   lets that review report
#   summary_replacements_a1.json    the proposed replacement summaries, held
#                                   apart from golden_dataset.json on purpose:
#                                   every recorded judge_score was measured
#                                   against the summaries as shipped, and
#                                   overwriting expected_summary takes its own
#                                   dated entry and a judge re-run
#
# gold_v2.json and adjudication_queue.csv appear when gold_v2.py has three
# sheets to work from. Neither exists yet, and gold_v2.py refuses to invent
# one from a single sheet.
