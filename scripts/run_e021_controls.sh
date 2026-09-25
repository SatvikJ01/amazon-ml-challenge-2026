#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
CAP="systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0"
$CAP .venv/bin/python -u -m src.train --tag trn2c_sib --exp E021c_stage3_no_sib --drop-features p2_rank n_anchors anchor_p_mean sib_name sib_addr sib_all sib_num_eq is_anchor; echo "EXIT $? control"
$CAP .venv/bin/python -u -m src.train --tag trn2c_sib --exp E021L_stage3_light --drop-features name_ratio name_partial name_tsort name_tset name_jw core_ratio core_tset core_partial nospace_partial nospace_lev core_nospace_partial skel_ratio skel_tset skel_nospace_partial addr_skel_tset addr_tset addr_tsort addr_partial addr_ratio name_idf_q name_idf_c name_idf_u addr_idf_q addr_idf_c addr_idf_u dig_jaccard dig_first_eq dig_any dig_n_q dig_n_c name_ntok_q name_ntok_c name_ntok_diff addr_ntok_c addr_empty_c name_nonascii_c digz_jaccard digz_first_eq digz_best_sim digz_first_sim s1_name_count s1_core_count cand_core_count_in_s1 name_len_ratio blk_score_gap blk_score_rank name_tset_gap name_tset_rank addr_tset_gap addr_tset_rank name_idf_u_gap name_idf_u_rank addr_idf_u_gap addr_idf_u_rank blk_score_src_gap n_cands; echo "EXIT $? light"
echo ALL_DONE
