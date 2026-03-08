Labels Backup
=============
Date: 2026-03-03

Reason: Spot-check of existing posture3 labels revealed ~40% error rate
(20 out of 50 random frames had wrong posture labels). Decided to relabel
all pig-present frames from scratch using the new depth-based prefilter
(threshold=1463mm, bar_margin=50px) which identifies 796 pig-present frames
out of 959 total.

Files:
  labels_posture3_backup_20260303.csv
    - Original posture3 labels (769 rows)
    - Contains mixed label scheme issue (label 3 from posture4 was remapped to 2)
    - Had labels for both pig-present AND no-pig frames
    - ~40% posture labels found incorrect during spot-check

  labels_posture3_clean_20260303.csv
    - Cleaned version (634 rows)
    - After label 3→2 remapping was applied

These files are kept for reference only. Do not use for training.
New labels will be created fresh in labels/labels_posture3.csv using
only the 796 pig-present frames from the depth prefilter.
