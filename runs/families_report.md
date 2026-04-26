# tagclean discover report

- centroids: `/home/synesis/tagclean/runs/bn_account_locked_qa_v1/stage2`
- thresholds: edge=0.88, pair=0.86, top_k=8, max_family_size=4
- multi-tag families: 97
- singletons: 154
- coverage: 461/1394

## Multi-tag families (sorted by min_edge_sim desc)

### `fam_3df0760e`  min=0.996  avg=0.996
- tags: smart_card_collection_fees, smart_card_fee
- row counts: {'smart_card_collection_fees': 182, 'smart_card_fee': 163}
- nearest excluded: [{'tag': 'card_lost_new_card_fee', 'min_sim': 0.9817, 'reason': 'covered_by_other_family'}, {'tag': 'smart_card_selected_but_get_pdf', 'min_sim': 0.9806, 'reason': 'non_reciprocal'}, {'tag': 'smart_card_collect_by_other', 'min_sim': 0.9794, 'reason': 'non_reciprocal'}]

### `fam_700c8b67`  min=0.996  avg=0.996
- tags: logout_available, logout_steps
- row counts: {'logout_available': 98, 'logout_steps': 98}
- nearest excluded: [{'tag': 'relogin_credentials_required', 'min_sim': 0.9777, 'reason': 'below_pair_threshold'}, {'tag': 'home_screen_usage', 'min_sim': 0.9754, 'reason': 'non_reciprocal'}, {'tag': 'login_safety_precautions', 'min_sim': 0.9751, 'reason': 'non_reciprocal'}]

### `fam_1f3499f4`  min=0.988  avg=0.991
- tags: card_damaged_how_to_get_new_card, nid_probashi_smart_card_06, smart_card_lost_how_to_get_new_one_again, smart_card_lost_reissue
- row counts: {'card_damaged_how_to_get_new_card': 170, 'nid_probashi_smart_card_06': 80, 'smart_card_lost_how_to_get_new_one_again': 189, 'smart_card_lost_reissue': 171}
- nearest excluded: [{'tag': 'voter_slip_lost_how_to_get_nid', 'min_sim': 0.9905, 'reason': 'covered_by_other_family'}, {'tag': 'card_lost_no_info_how_to_get_new', 'min_sim': 0.9896, 'reason': 'below_pair_threshold'}, {'tag': 'nid_fee_09', 'min_sim': 0.9894, 'reason': 'below_pair_threshold'}]

### `fam_d852a543`  min=0.987  avg=0.989
- tags: fingerprint_update_applied_not_approved, nid_biometric_update_status_01, nid_biometric_update_status_02
- row counts: {'fingerprint_update_applied_not_approved': 185, 'nid_biometric_update_status_01': 81, 'nid_biometric_update_status_02': 81}
- nearest excluded: [{'tag': 'finderprint_adjudication_pending', 'min_sim': 0.988, 'reason': 'below_pair_threshold'}, {'tag': 'nid_biometric_update_not_working_01', 'min_sim': 0.9874, 'reason': 'below_pair_threshold'}, {'tag': 'fingerprint_error_info_not_found', 'min_sim': 0.9863, 'reason': 'covered_by_other_family'}]

### `fam_c8a0fa16`  min=0.987  avg=0.987
- tags: pre_use_privacy_policy_read, preparation_before_app_use
- row counts: {'pre_use_privacy_policy_read': 101, 'preparation_before_app_use': 96}
- nearest excluded: [{'tag': 'instruction_following_tips', 'min_sim': 0.9824, 'reason': 'covered_by_other_family'}, {'tag': 'app_requirements_internet_gps_vpn', 'min_sim': 0.9814, 'reason': 'covered_by_other_family'}, {'tag': 'process_requirements', 'min_sim': 0.9798, 'reason': 'covered_by_other_family'}]

### `fam_a8d130b5`  min=0.986  avg=0.986
- tags: online_apply, online_no_apply
- row counts: {'online_apply': 183, 'online_no_apply': 192}
- nearest excluded: [{'tag': 'online_card_correction', 'min_sim': 0.9802, 'reason': 'covered_by_other_family'}, {'tag': 'online_portal_registration', 'min_sim': 0.9776, 'reason': 'covered_by_other_family'}, {'tag': 'how_to_apply_for_correction_in_online_portal', 'min_sim': 0.9766, 'reason': 'covered_by_other_family'}]

### `fam_dc8c4128`  min=0.986  avg=0.989
- tags: help_center_access, help_center_definition, problem_help_suggestion, when_to_seek_help
- row counts: {'help_center_access': 99, 'help_center_definition': 95, 'problem_help_suggestion': 98, 'when_to_seek_help': 99}
- nearest excluded: [{'tag': 'help_center_contents', 'min_sim': 0.9941, 'reason': 'below_pair_threshold'}, {'tag': 'help_contact_location', 'min_sim': 0.9886, 'reason': 'covered_by_other_family'}, {'tag': 'app_not_opening_troubleshoot', 'min_sim': 0.9861, 'reason': 'covered_by_other_family'}]

### `fam_2bbc0469`  min=0.986  avg=0.990
- tags: card_correction_request_canceled_by_consumer_regarding_refund_fees, card_correction_request_failed_query_regarding_refund_fees, card_correction_request_failed_query_regarding_repayment_fees, correction_application_rejected_need_new_fee
- row counts: {'card_correction_request_canceled_by_consumer_regarding_refund_fees': 173, 'card_correction_request_failed_query_regarding_refund_fees': 170, 'card_correction_request_failed_query_regarding_repayment_fees': 178, 'correction_application_rejected_need_new_fee': 167}
- nearest excluded: [{'tag': 'softwere_bug_in_card_processing_system', 'min_sim': 0.9844, 'reason': 'below_pair_threshold'}, {'tag': 'nid_fee_wrong_nid_refund_01', 'min_sim': 0.9795, 'reason': 'below_pair_threshold'}, {'tag': 'card_correction_fees', 'min_sim': 0.9786, 'reason': 'below_pair_threshold'}]

### `fam_648f7e45`  min=0.985  avg=0.991
- tags: forgot_password_flow, password_reset_required_info, post_registration_password_reset, show_password
- row counts: {'forgot_password_flow': 97, 'password_reset_required_info': 98, 'post_registration_password_reset': 94, 'show_password': 101}
- nearest excluded: [{'tag': 'password_requirements', 'min_sim': 0.9887, 'reason': 'covered_by_other_family'}, {'tag': 'password_change_process', 'min_sim': 0.9876, 'reason': 'below_pair_threshold'}, {'tag': 'password_change_frequency', 'min_sim': 0.9857, 'reason': 'covered_by_other_family'}]

### `fam_5a29f324`  min=0.985  avg=0.991
- tags: fingerprint_error_info_not_found, nid_afis_04, nid_fingerprint_mismatch_sim_issue_01, nid_missing_fingerprint_sim_issue_01
- row counts: {'fingerprint_error_info_not_found': 169, 'nid_afis_04': 75, 'nid_fingerprint_mismatch_sim_issue_01': 72, 'nid_missing_fingerprint_sim_issue_01': 72}
- nearest excluded: [{'tag': 'nid_afis_01', 'min_sim': 0.9931, 'reason': 'covered_by_other_family'}, {'tag': 'nid_biometric_update_not_working_01', 'min_sim': 0.9919, 'reason': 'below_pair_threshold'}, {'tag': 'fingerprint_update_applied_not_approved', 'min_sim': 0.9863, 'reason': 'covered_by_other_family'}]

### `fam_e1f94cc7`  min=0.985  avg=0.987
- tags: foreign_resident_card_collection, nid_probashi_smart_card_03, smart_card_from_where
- row counts: {'foreign_resident_card_collection': 178, 'nid_probashi_smart_card_03': 80, 'smart_card_from_where': 174}
- nearest excluded: [{'tag': 'not_getting_smart_card_though_newer_voter_getting', 'min_sim': 0.9908, 'reason': 'below_pair_threshold'}, {'tag': 'nid_probashi_smart_card_09', 'min_sim': 0.9878, 'reason': 'covered_by_other_family'}, {'tag': 'wish_to_create_nid_smart_card_new', 'min_sim': 0.9847, 'reason': 'below_pair_threshold'}]

### `fam_a88f409a`  min=0.985  avg=0.991
- tags: closing_statement, goodbye, request_for_clear_pronunciation, request_for_feedback_or_suggestions
- row counts: {'closing_statement': 173, 'goodbye': 226, 'request_for_clear_pronunciation': 188, 'request_for_feedback_or_suggestions': 184}
- nearest excluded: [{'tag': 'fraction', 'min_sim': 0.9913, 'reason': 'covered_by_other_family'}, {'tag': 'repeat_again', 'min_sim': 0.991, 'reason': 'below_pair_threshold'}, {'tag': 'warning_to_prank_caller', 'min_sim': 0.9874, 'reason': 'covered_by_other_family'}]

### `fam_14ae6169`  min=0.984  avg=0.987
- tags: verification_completion_notification, verification_in_progress_app_close, verification_in_progress_wait, verification_time
- row counts: {'verification_completion_notification': 101, 'verification_in_progress_app_close': 101, 'verification_in_progress_wait': 100, 'verification_time': 100}
- nearest excluded: [{'tag': 'post_submission_verification', 'min_sim': 0.9845, 'reason': 'below_pair_threshold'}, {'tag': 'unable_to_answer', 'min_sim': 0.9811, 'reason': 'non_reciprocal'}, {'tag': 'correction_application_status', 'min_sim': 0.9799, 'reason': 'non_reciprocal'}]

### `fam_6845800d`  min=0.984  avg=0.988
- tags: photo_done_years_ago_but_no_card, picture_done_but_lost_or_no_sms_slip, voter_slip_lost_how_to_get_new_card, voter_slip_lost_how_to_get_nid
- row counts: {'photo_done_years_ago_but_no_card': 176, 'picture_done_but_lost_or_no_sms_slip': 210, 'voter_slip_lost_how_to_get_new_card': 181, 'voter_slip_lost_how_to_get_nid': 161}
- nearest excluded: [{'tag': 'card_lost_no_info_how_to_get_new', 'min_sim': 0.9928, 'reason': 'below_pair_threshold'}, {'tag': 'not_getting_smart_card_though_newer_voter_getting', 'min_sim': 0.9922, 'reason': 'below_pair_threshold'}, {'tag': 'card_damaged_how_to_get_new_card', 'min_sim': 0.9905, 'reason': 'covered_by_other_family'}]

### `fam_3678a9b9`  min=0.983  avg=0.985
- tags: foreign_resident_card_registration_process, foreign_resident_card_registration_query_country_limitation, nrb_application_status, unable_to_answer
- row counts: {'foreign_resident_card_registration_process': 196, 'foreign_resident_card_registration_query_country_limitation': 194, 'nrb_application_status': 186, 'unable_to_answer': 1986}
- nearest excluded: [{'tag': 'nid_new_registration_04', 'min_sim': 0.9891, 'reason': 'covered_by_other_family'}, {'tag': 'foreign_resident_card_registration_required_documents', 'min_sim': 0.9883, 'reason': 'below_pair_threshold'}, {'tag': 'nid_dual_citizenship_02', 'min_sim': 0.9862, 'reason': 'below_pair_threshold'}]

### `fam_9dcd16fc`  min=0.983  avg=0.987
- tags: card_information_correction, how_to_apply_for_correction_in_online_portal, online_card_correction, online_card_correction_application_done_now_what
- row counts: {'card_information_correction': 172, 'how_to_apply_for_correction_in_online_portal': 179, 'online_card_correction': 177, 'online_card_correction_application_done_now_what': 172}
- nearest excluded: [{'tag': 'nid_information_misentry_correction_procedure', 'min_sim': 0.9881, 'reason': 'covered_by_other_family'}, {'tag': 'nid_probashi_registration_02', 'min_sim': 0.9873, 'reason': 'covered_by_other_family'}, {'tag': 'wrong_information_in_online_application_what_to_do', 'min_sim': 0.9868, 'reason': 'below_pair_threshold'}]

### `fam_74518ac7`  min=0.983  avg=0.983
- tags: nid_correction_online_01, nid_religion_change_documents_01
- row counts: {'nid_correction_online_01': 81, 'nid_religion_change_documents_01': 81}
- nearest excluded: [{'tag': 'name_correction_in_nid_card', 'min_sim': 0.9766, 'reason': 'covered_by_other_family'}, {'tag': 'card_information_correction', 'min_sim': 0.9763, 'reason': 'non_reciprocal'}, {'tag': 'birthplace_correction_new', 'min_sim': 0.9743, 'reason': 'non_reciprocal'}]

### `fam_81904d01`  min=0.983  avg=0.985
- tags: instruction_following_tips, login_safety_precautions, security_best_practices, voting_precautions
- row counts: {'instruction_following_tips': 101, 'login_safety_precautions': 98, 'security_best_practices': 97, 'voting_precautions': 101}
- nearest excluded: [{'tag': 'login_credentials_required', 'min_sim': 0.9891, 'reason': 'below_pair_threshold'}, {'tag': 'app_requirements_internet_gps_vpn', 'min_sim': 0.9862, 'reason': 'covered_by_other_family'}, {'tag': 'problem_help_suggestion', 'min_sim': 0.9856, 'reason': 'covered_by_other_family'}]

### `fam_82170516`  min=0.983  avg=0.983
- tags: nid_address_update_05, nid_new_registration_01
- row counts: {'nid_address_update_05': 81, 'nid_new_registration_01': 81}
- nearest excluded: [{'tag': 'necessary_documents_to_modify_voter_area', 'min_sim': 0.9771, 'reason': 'below_pair_threshold'}, {'tag': 'voter_area_change', 'min_sim': 0.976, 'reason': 'non_reciprocal'}, {'tag': 'foreign_resident_card_registration_required_documents', 'min_sim': 0.974, 'reason': 'non_reciprocal'}]

### `fam_a6e35627`  min=0.983  avg=0.986
- tags: nid_probashi_smart_card_02, nid_probashi_smart_card_04, nid_probashi_smart_card_08, nid_probashi_smart_card_09
- row counts: {'nid_probashi_smart_card_02': 81, 'nid_probashi_smart_card_04': 80, 'nid_probashi_smart_card_08': 80, 'nid_probashi_smart_card_09': 78}
- nearest excluded: [{'tag': 'smart_card_collect_by_other', 'min_sim': 0.9901, 'reason': 'below_pair_threshold'}, {'tag': 'foreign_resident_card_collection', 'min_sim': 0.9878, 'reason': 'covered_by_other_family'}, {'tag': 'nrb_application_status', 'min_sim': 0.9842, 'reason': 'covered_by_other_family'}]

### `fam_9a18e967`  min=0.983  avg=0.989
- tags: account_locked, account_locked_retrials, account_locked_unlock_request, eight_hours_time_up
- row counts: {'account_locked': 183, 'account_locked_retrials': 174, 'account_locked_unlock_request': 182, 'eight_hours_time_up': 184}
- nearest excluded: [{'tag': 'information_correct_but_account_locked', 'min_sim': 0.9876, 'reason': 'below_pair_threshold'}, {'tag': 'no_otp_in_registration', 'min_sim': 0.9758, 'reason': 'non_reciprocal'}, {'tag': 'card_lock', 'min_sim': 0.9755, 'reason': 'below_pair_threshold'}]

### `fam_5a81a315`  min=0.983  avg=0.989
- tags: nid_correction_category_approver_01, nid_correction_category_approver_02, nid_correction_category_approver_03, nid_correction_category_approver_04
- row counts: {'nid_correction_category_approver_01': 82, 'nid_correction_category_approver_02': 80, 'nid_correction_category_approver_03': 75, 'nid_correction_category_approver_04': 80}
- nearest excluded: [{'tag': 'nid_correction_category_approver_07', 'min_sim': 0.9853, 'reason': 'below_pair_threshold'}, {'tag': 'nid_correction_category_approver_06', 'min_sim': 0.9851, 'reason': 'below_pair_threshold'}, {'tag': 'nid_correction_category_approver_05', 'min_sim': 0.9812, 'reason': 'below_pair_threshold'}]

### `fam_c5880397`  min=0.983  avg=0.988
- tags: correction_application_current_status, correction_application_status, correction_applied_more_correction_needed, nid_correction_online_06
- row counts: {'correction_application_current_status': 162, 'correction_application_status': 157, 'correction_applied_more_correction_needed': 171, 'nid_correction_online_06': 78}
- nearest excluded: [{'tag': 'wrong_information_in_online_application_what_to_do', 'min_sim': 0.9871, 'reason': 'below_pair_threshold'}, {'tag': 'photo_signature_change_application_not_approved', 'min_sim': 0.9846, 'reason': 'below_pair_threshold'}, {'tag': 'unable_to_answer', 'min_sim': 0.9836, 'reason': 'covered_by_other_family'}]

### `fam_2c865946`  min=0.982  avg=0.984
- tags: password_sharing_warning, strong_password_importance, weak_password_risk
- row counts: {'password_sharing_warning': 101, 'strong_password_importance': 100, 'weak_password_risk': 101}
- nearest excluded: [{'tag': 'password_requirements', 'min_sim': 0.9899, 'reason': 'covered_by_other_family'}, {'tag': 'show_password', 'min_sim': 0.9851, 'reason': 'covered_by_other_family'}, {'tag': 'password_character_requirements', 'min_sim': 0.9851, 'reason': 'covered_by_other_family'}]

### `fam_88ce2b71`  min=0.982  avg=0.982
- tags: otp_delivery_time, password_reset_time
- row counts: {'otp_delivery_time': 101, 'password_reset_time': 101}
- nearest excluded: [{'tag': 'password_reset_required_info', 'min_sim': 0.9789, 'reason': 'non_reciprocal'}, {'tag': 'otp_length', 'min_sim': 0.978, 'reason': 'covered_by_other_family'}, {'tag': 'otp_not_received', 'min_sim': 0.9774, 'reason': 'non_reciprocal'}]

### `fam_7faf5985`  min=0.982  avg=0.982
- tags: app_requirements_internet_gps_vpn, location_service_required_reason
- row counts: {'app_requirements_internet_gps_vpn': 99, 'location_service_required_reason': 101}
- nearest excluded: [{'tag': 'process_requirements', 'min_sim': 0.9871, 'reason': 'covered_by_other_family'}, {'tag': 'instruction_following_tips', 'min_sim': 0.9862, 'reason': 'covered_by_other_family'}, {'tag': 'app_not_opening_troubleshoot', 'min_sim': 0.9852, 'reason': 'covered_by_other_family'}]

### `fam_e3fc8441`  min=0.982  avg=0.985
- tags: address_change_card_download_process, no_download_option_after_correction, previous_information_in_downloaded_card_after_correction
- row counts: {'address_change_card_download_process': 172, 'no_download_option_after_correction': 171, 'previous_information_in_downloaded_card_after_correction': 172}
- nearest excluded: [{'tag': 'card_avail_after_address_change', 'min_sim': 0.9839, 'reason': 'covered_by_other_family'}, {'tag': 'updated_nid_after_correction', 'min_sim': 0.9837, 'reason': 'below_pair_threshold'}, {'tag': 'new_voter_can_not_download_card_with_form_number', 'min_sim': 0.9811, 'reason': 'below_pair_threshold'}]

### `fam_67ebd19d`  min=0.982  avg=0.982
- tags: pre_voting_video_mandatory, video_tutorial_why
- row counts: {'pre_voting_video_mandatory': 101, 'video_tutorial_why': 100}
- nearest excluded: [{'tag': 'video_end_next_step', 'min_sim': 0.9748, 'reason': 'below_pair_threshold'}, {'tag': 'instruction_following_tips', 'min_sim': 0.9702, 'reason': 'non_reciprocal'}, {'tag': 'help_center_definition', 'min_sim': 0.9687, 'reason': 'non_reciprocal'}]

### `fam_1746939b`  min=0.981  avg=0.986
- tags: app_not_opening_troubleshoot, login_page_not_opening, post_login_page_not_loading
- row counts: {'app_not_opening_troubleshoot': 101, 'login_page_not_opening': 98, 'post_login_page_not_loading': 100}
- nearest excluded: [{'tag': 'problem_help_suggestion', 'min_sim': 0.9861, 'reason': 'covered_by_other_family'}, {'tag': 'otp_not_received', 'min_sim': 0.9854, 'reason': 'covered_by_other_family'}, {'tag': 'nid_verification_failed', 'min_sim': 0.9852, 'reason': 'below_pair_threshold'}]

### `fam_14a6fa1e`  min=0.981  avg=0.985
- tags: nid_address_update_12, nid_voter_area_transfer_02, voter_area_change, voter_area_query
- row counts: {'nid_address_update_12': 81, 'nid_voter_area_transfer_02': 78, 'voter_area_change': 171, 'voter_area_query': 181}
- nearest excluded: [{'tag': 'no_own_house_in_voter_area', 'min_sim': 0.9876, 'reason': 'covered_by_other_family'}, {'tag': 'nid_voter_area_transfer_06', 'min_sim': 0.987, 'reason': 'covered_by_other_family'}, {'tag': 'reigister_as_new_voter_but_birth_registration_different_address', 'min_sim': 0.9855, 'reason': 'covered_by_other_family'}]

### `fam_18f4419e`  min=0.981  avg=0.981
- tags: card_correction_one_information_maximum_limit, nid_reissue_08
- row counts: {'card_correction_one_information_maximum_limit': 182, 'nid_reissue_08': 83}
- nearest excluded: [{'tag': 'correction_application_current_status', 'min_sim': 0.9793, 'reason': 'non_reciprocal'}, {'tag': 'correction_applied_more_correction_needed', 'min_sim': 0.979, 'reason': 'covered_by_other_family'}, {'tag': 'correction_application_status', 'min_sim': 0.9784, 'reason': 'non_reciprocal'}]

### `fam_6367a366`  min=0.980  avg=0.980
- tags: reissue_supporting_document, voter_card_damaged
- row counts: {'reissue_supporting_document': 184, 'voter_card_damaged': 175}
- nearest excluded: [{'tag': 'nid_reissue_07', 'min_sim': 0.9815, 'reason': 'covered_by_other_family'}, {'tag': 'nid_reissue_09', 'min_sim': 0.9814, 'reason': 'below_pair_threshold'}, {'tag': 'tin_number_correct_or_add_document', 'min_sim': 0.9793, 'reason': 'below_pair_threshold'}]

### `fam_1e986aa6`  min=0.980  avg=0.982
- tags: smart_card_printing_status_sms_how, smart_card_ready_sms, smart_card_status
- row counts: {'smart_card_printing_status_sms_how': 182, 'smart_card_ready_sms': 174, 'smart_card_status': 191}
- nearest excluded: [{'tag': 'nid_smart_card_print_status_01', 'min_sim': 0.9874, 'reason': 'covered_by_other_family'}, {'tag': 'smart_card_selected_but_get_pdf', 'min_sim': 0.9845, 'reason': 'below_pair_threshold'}, {'tag': 'nid_probashi_smart_card_06', 'min_sim': 0.9821, 'reason': 'non_reciprocal'}]

### `fam_ca01692f`  min=0.980  avg=0.986
- tags: double_voter_registration_issue, nid_match_found_verification_failed_how_to_solve, nid_new_registration_04
- row counts: {'double_voter_registration_issue': 176, 'nid_match_found_verification_failed_how_to_solve': 167, 'nid_new_registration_04': 78}
- nearest excluded: [{'tag': 'unable_to_answer', 'min_sim': 0.9891, 'reason': 'covered_by_other_family'}, {'tag': 'nid_adjudication_pending', 'min_sim': 0.9842, 'reason': 'below_pair_threshold'}, {'tag': 'nid_biometric_update_not_working_01', 'min_sim': 0.9819, 'reason': 'below_pair_threshold'}]

### `fam_dbc43c5c`  min=0.980  avg=0.985
- tags: sem_app_info_01, sem_app_info_04, sem_app_info_06, sem_app_info_11
- row counts: {'sem_app_info_01': 80, 'sem_app_info_04': 81, 'sem_app_info_06': 81, 'sem_app_info_11': 81}
- nearest excluded: [{'tag': 'sem_app_info_10', 'min_sim': 0.9889, 'reason': 'below_pair_threshold'}, {'tag': 'sem_app_info_09', 'min_sim': 0.9812, 'reason': 'below_pair_threshold'}, {'tag': 'sem_app_info_03', 'min_sim': 0.9811, 'reason': 'below_pair_threshold'}]

### `fam_ad5bf92c`  min=0.980  avg=0.980
- tags: new_voter_how_to_get_card, online_new_voter_registration
- row counts: {'new_voter_how_to_get_card': 168, 'online_new_voter_registration': 172}
- nearest excluded: [{'tag': 'new_voter_smart_card_how_to_get', 'min_sim': 0.9902, 'reason': 'covered_by_other_family'}, {'tag': 'new_voter_got_normal_card_when_get_smart_card', 'min_sim': 0.9828, 'reason': 'below_pair_threshold'}, {'tag': 'wish_to_create_nid_smart_card_new', 'min_sim': 0.9801, 'reason': 'non_reciprocal'}]

### `fam_0845a5c9`  min=0.980  avg=0.983
- tags: incorrect_info_restart, otp_not_received, wrong_mobile_number_consequence, wrong_otp_registration
- row counts: {'incorrect_info_restart': 100, 'otp_not_received': 100, 'wrong_mobile_number_consequence': 99, 'wrong_otp_registration': 97}
- nearest excluded: [{'tag': 'wrong_info_penalty', 'min_sim': 0.9882, 'reason': 'below_pair_threshold'}, {'tag': 'nid_verification_failed', 'min_sim': 0.9856, 'reason': 'below_pair_threshold'}, {'tag': 'app_not_opening_troubleshoot', 'min_sim': 0.9854, 'reason': 'covered_by_other_family'}]

### `fam_ff0308e8`  min=0.979  avg=0.981
- tags: birth_certificate_number_correct_or_add_document_new, cell_phone_number_correct_or_add_document_new, parent_spouse_name_correct_or_add_document_new, passport_number_correct_or_add_document_new
- row counts: {'birth_certificate_number_correct_or_add_document_new': 185, 'cell_phone_number_correct_or_add_document_new': 191, 'parent_spouse_name_correct_or_add_document_new': 228, 'passport_number_correct_or_add_document_new': 186}
- nearest excluded: [{'tag': 'birthplace_correction_new', 'min_sim': 0.9856, 'reason': 'covered_by_other_family'}, {'tag': 'spouse_name_correction_new', 'min_sim': 0.9831, 'reason': 'covered_by_other_family'}, {'tag': 'parents_name_correction_new', 'min_sim': 0.9817, 'reason': 'covered_by_other_family'}]

### `fam_b46e07bf`  min=0.979  avg=0.979
- tags: nid_vote_center_change_01, no_own_house_in_voter_area
- row counts: {'nid_vote_center_change_01': 81, 'no_own_house_in_voter_area': 172}
- nearest excluded: [{'tag': 'voter_area_change', 'min_sim': 0.9876, 'reason': 'covered_by_other_family'}, {'tag': 'new_voter_pre_condition', 'min_sim': 0.9836, 'reason': 'below_pair_threshold'}, {'tag': 'father_and_my_adress_different', 'min_sim': 0.9812, 'reason': 'below_pair_threshold'}]

### `fam_98d2eb5b`  min=0.979  avg=0.979
- tags: policy_menu_contents, privacy_policy_menu_location
- row counts: {'policy_menu_contents': 101, 'privacy_policy_menu_location': 101}
- nearest excluded: [{'tag': 'notification_menu_contents', 'min_sim': 0.9761, 'reason': 'below_pair_threshold'}, {'tag': 'profile_menu_contents', 'min_sim': 0.9747, 'reason': 'below_pair_threshold'}, {'tag': 'notification_screen_info', 'min_sim': 0.9741, 'reason': 'below_pair_threshold'}]

### `fam_519e3017`  min=0.978  avg=0.985
- tags: address_change_sms_to_mobile_how_many_days_to_get_approved, nid_address_update_08, permanent_address_change_process_duration, present_address_change_process_duration
- row counts: {'address_change_sms_to_mobile_how_many_days_to_get_approved': 171, 'nid_address_update_08': 81, 'permanent_address_change_process_duration': 169, 'present_address_change_process_duration': 175}
- nearest excluded: [{'tag': 'permanent_address_change_fees', 'min_sim': 0.977, 'reason': 'below_pair_threshold'}, {'tag': 'unable_to_answer', 'min_sim': 0.9762, 'reason': 'non_reciprocal'}, {'tag': 'present_address_change_procedure', 'min_sim': 0.9748, 'reason': 'non_reciprocal'}]

### `fam_5d162440`  min=0.978  avg=0.985
- tags: ballot_tracking_definition, ballot_tracking_steps, tracking_status_meaning, vote_tracking_definition
- row counts: {'ballot_tracking_definition': 91, 'ballot_tracking_steps': 100, 'tracking_status_meaning': 100, 'vote_tracking_definition': 97}
- nearest excluded: [{'tag': 'ballot_tracking_button_activation', 'min_sim': 0.989, 'reason': 'below_pair_threshold'}, {'tag': 'verification_time', 'min_sim': 0.9791, 'reason': 'covered_by_other_family'}, {'tag': 'instruction_following_tips', 'min_sim': 0.979, 'reason': 'non_reciprocal'}]

### `fam_c3f3e008`  min=0.978  avg=0.982
- tags: how_to_complain_not_getting_help_from_election_office, new_voter_application_not_submitted_to_upazila_or_thana_office, nid_bribery_complaint_01
- row counts: {'how_to_complain_not_getting_help_from_election_office': 171, 'new_voter_application_not_submitted_to_upazila_or_thana_office': 173, 'nid_bribery_complaint_01': 81}
- nearest excluded: [{'tag': 'unable_to_answer', 'min_sim': 0.9859, 'reason': 'covered_by_other_family'}, {'tag': 'new_voter_no_messages_from_upzila_or_thana', 'min_sim': 0.9854, 'reason': 'covered_by_other_family'}, {'tag': 'new_voter_application_cancelled', 'min_sim': 0.9852, 'reason': 'covered_by_other_family'}]

### `fam_cb5e9d1e`  min=0.978  avg=0.982
- tags: nid_appeal_disposal_time_01, nid_application_rejection_notice_time_01, nid_application_verification_time_01
- row counts: {'nid_appeal_disposal_time_01': 80, 'nid_application_rejection_notice_time_01': 81, 'nid_application_verification_time_01': 79}
- nearest excluded: [{'tag': 'nid_card_validity_period_01', 'min_sim': 0.9752, 'reason': 'below_pair_threshold'}, {'tag': 'correction_application_status', 'min_sim': 0.9725, 'reason': 'non_reciprocal'}, {'tag': 'online_application_pending', 'min_sim': 0.9725, 'reason': 'non_reciprocal'}]

### `fam_f2a9433a`  min=0.978  avg=0.982
- tags: nid_wallet_download, nid_wallet_download_alternative, nid_wallet_download_iphone, no_android_phone
- row counts: {'nid_wallet_download': 170, 'nid_wallet_download_alternative': 169, 'nid_wallet_download_iphone': 176, 'no_android_phone': 172}
- nearest excluded: [{'tag': 'no_download_option_after_correction', 'min_sim': 0.9732, 'reason': 'non_reciprocal'}, {'tag': 'correction_fee_submitted_but_balance_zero', 'min_sim': 0.9704, 'reason': 'non_reciprocal'}, {'tag': 'no_face_verification', 'min_sim': 0.9665, 'reason': 'non_reciprocal'}]

### `fam_e4c52104`  min=0.978  avg=0.982
- tags: foreign_resident_card_picture_done__inquery_done_no_msg_new, foreign_resident_card_picture_done_not_inquery_from_upzilla_new, nid_probashi_biometric_status_05
- row counts: {'foreign_resident_card_picture_done__inquery_done_no_msg_new': 183, 'foreign_resident_card_picture_done_not_inquery_from_upzilla_new': 177, 'nid_probashi_biometric_status_05': 81}
- nearest excluded: [{'tag': 'nid_probashi_application_rejected_01', 'min_sim': 0.9786, 'reason': 'covered_by_other_family'}, {'tag': 'nrb_application_status', 'min_sim': 0.9786, 'reason': 'non_reciprocal'}, {'tag': 'new_voter_no_messages_from_upzila_or_thana', 'min_sim': 0.9782, 'reason': 'covered_by_other_family'}]

### `fam_11d292fd`  min=0.978  avg=0.986
- tags: password_change_frequency, password_character_requirements, password_min_length, password_requirements
- row counts: {'password_change_frequency': 101, 'password_character_requirements': 99, 'password_min_length': 100, 'password_requirements': 96}
- nearest excluded: [{'tag': 'strong_password_importance', 'min_sim': 0.9899, 'reason': 'covered_by_other_family'}, {'tag': 'password_reset_required_info', 'min_sim': 0.9887, 'reason': 'covered_by_other_family'}, {'tag': 'show_password', 'min_sim': 0.9866, 'reason': 'covered_by_other_family'}]

### `fam_9e3c9581`  min=0.978  avg=0.985
- tags: card_avail_after_address_change, permanent_address_new, present_address_change_procedure, present_and_permanent_address_generic_new
- row counts: {'card_avail_after_address_change': 166, 'permanent_address_new': 183, 'present_address_change_procedure': 192, 'present_and_permanent_address_generic_new': 180}
- nearest excluded: [{'tag': 'address_change_card_download_process', 'min_sim': 0.9839, 'reason': 'covered_by_other_family'}, {'tag': 'card_damaged_how_to_get_new_card', 'min_sim': 0.9836, 'reason': 'non_reciprocal'}, {'tag': 'updated_nid_after_correction', 'min_sim': 0.9805, 'reason': 'non_reciprocal'}]

### `fam_47a490ca`  min=0.978  avg=0.984
- tags: card_lost_and_damaged_cost, card_lost_new_card_fee, nid_reregistration_fee_01
- row counts: {'card_lost_and_damaged_cost': 164, 'card_lost_new_card_fee': 167, 'nid_reregistration_fee_01': 81}
- nearest excluded: [{'tag': 'voter_slip_lost_how_to_get_nid', 'min_sim': 0.9831, 'reason': 'non_reciprocal'}, {'tag': 'smart_card_fee', 'min_sim': 0.9817, 'reason': 'covered_by_other_family'}, {'tag': 'card_damaged_how_to_get_new_card', 'min_sim': 0.9803, 'reason': 'non_reciprocal'}]

### `fam_06167204`  min=0.978  avg=0.984
- tags: address_change_online_new, permanent_address_update_procedure, persent_address_correction_procedure, together_present_and_permanent_address_change
- row counts: {'address_change_online_new': 193, 'permanent_address_update_procedure': 170, 'persent_address_correction_procedure': 167, 'together_present_and_permanent_address_change': 172}
- nearest excluded: [{'tag': 'present_address_change_procedure', 'min_sim': 0.9793, 'reason': 'covered_by_other_family'}, {'tag': 'present_and_permanent_address_generic_new', 'min_sim': 0.9765, 'reason': 'covered_by_other_family'}, {'tag': 'nid_address_update_13', 'min_sim': 0.9756, 'reason': 'below_pair_threshold'}]

### `fam_f4500dc7`  min=0.978  avg=0.978
- tags: nid_voter_area_transfer_05, voter_area_change_ongoing
- row counts: {'nid_voter_area_transfer_05': 82, 'voter_area_change_ongoing': 172}
- nearest excluded: [{'tag': 'voter_area_change_not_reflected', 'min_sim': 0.9866, 'reason': 'below_pair_threshold'}, {'tag': 'correction_application_current_status', 'min_sim': 0.9828, 'reason': 'covered_by_other_family'}, {'tag': 'correction_application_status', 'min_sim': 0.9823, 'reason': 'covered_by_other_family'}]

### `fam_a216e408`  min=0.977  avg=0.983
- tags: dead_people_id_card_update, nid_misc_04, parents_or_spouse_death_related_documents_to_declare_death, wrongly_dead_declared
- row counts: {'dead_people_id_card_update': 167, 'nid_misc_04': 80, 'parents_or_spouse_death_related_documents_to_declare_death': 193, 'wrongly_dead_declared': 196}
- nearest excluded: [{'tag': 'nid_misc_05', 'min_sim': 0.9848, 'reason': 'below_pair_threshold'}, {'tag': 'parents_name_correction_new', 'min_sim': 0.9809, 'reason': 'covered_by_other_family'}, {'tag': 'parent_spouse_name_correct_or_add_document_new', 'min_sim': 0.9809, 'reason': 'covered_by_other_family'}]

### `fam_0ba31774`  min=0.977  avg=0.977
- tags: form_number_not_working, nid_new_registration_03
- row counts: {'form_number_not_working': 173, 'nid_new_registration_03': 81}
- nearest excluded: [{'tag': 'nid_biometric_update_not_working_01', 'min_sim': 0.9744, 'reason': 'non_reciprocal'}, {'tag': 'unable_to_answer', 'min_sim': 0.9743, 'reason': 'non_reciprocal'}, {'tag': 'nid_new_registration_04', 'min_sim': 0.9742, 'reason': 'non_reciprocal'}]

### `fam_f008a65c`  min=0.977  avg=0.977
- tags: call_center_bengali_support, help_contact_location
- row counts: {'call_center_bengali_support': 100, 'help_contact_location': 91}
- nearest excluded: [{'tag': 'help_center_access', 'min_sim': 0.9886, 'reason': 'covered_by_other_family'}, {'tag': 'problem_help_suggestion', 'min_sim': 0.9865, 'reason': 'covered_by_other_family'}, {'tag': 'help_center_definition', 'min_sim': 0.9859, 'reason': 'covered_by_other_family'}]

### `fam_522297bd`  min=0.977  avg=0.980
- tags: foreign_resident_multiple_registration_new, nid_dual_citizenship_03, nid_postal_vote_abroad_01, nid_probashi_voter_list_01
- row counts: {'foreign_resident_multiple_registration_new': 183, 'nid_dual_citizenship_03': 81, 'nid_postal_vote_abroad_01': 81, 'nid_probashi_voter_list_01': 81}
- nearest excluded: [{'tag': 'condition_for_foreign_voter', 'min_sim': 0.9857, 'reason': 'below_pair_threshold'}, {'tag': 'foreign_resident_card_registration_process', 'min_sim': 0.9849, 'reason': 'covered_by_other_family'}, {'tag': 'nid_probashi_address_requirement_01', 'min_sim': 0.977, 'reason': 'covered_by_other_family'}]

### `fam_7b685a71`  min=0.977  avg=0.981
- tags: educational_qualification_correction, nid_correction_online_08, profession_change
- row counts: {'educational_qualification_correction': 183, 'nid_correction_online_08': 79, 'profession_change': 181}
- nearest excluded: [{'tag': 'card_information_correction', 'min_sim': 0.9835, 'reason': 'non_reciprocal'}, {'tag': 'nid_information_misentry_correction_procedure', 'min_sim': 0.9809, 'reason': 'covered_by_other_family'}, {'tag': 'ssc_certificate_for_correction', 'min_sim': 0.979, 'reason': 'covered_by_other_family'}]

### `fam_bb4bfda3`  min=0.977  avg=0.982
- tags: card_reissue_fees, nid_reissue_02, nid_reissue_03
- row counts: {'card_reissue_fees': 182, 'nid_reissue_02': 81, 'nid_reissue_03': 80}
- nearest excluded: [{'tag': 'correction_fee_payment', 'min_sim': 0.9781, 'reason': 'non_reciprocal'}, {'tag': 'nid_reissue_06', 'min_sim': 0.9777, 'reason': 'below_pair_threshold'}, {'tag': 'card_correction_request_canceled_by_consumer_regarding_refund_fees', 'min_sim': 0.9762, 'reason': 'covered_by_other_family'}]

### `fam_8e410f91`  min=0.977  avg=0.981
- tags: new_voter_application_cancelled, new_voter_no_messages_from_upzila_or_thana, nid_afis_02, voter_halnagad_registration_not_done
- row counts: {'new_voter_application_cancelled': 176, 'new_voter_no_messages_from_upzila_or_thana': 179, 'nid_afis_02': 81, 'voter_halnagad_registration_not_done': 177}
- nearest excluded: [{'tag': 'nid_afis_03', 'min_sim': 0.9868, 'reason': 'below_pair_threshold'}, {'tag': 'new_voter_application_not_submitted_to_upazila_or_thana_office', 'min_sim': 0.9854, 'reason': 'covered_by_other_family'}, {'tag': 'unable_to_answer', 'min_sim': 0.9847, 'reason': 'covered_by_other_family'}]

### `fam_1502cc86`  min=0.977  avg=0.977
- tags: nid_new_registration_06, sem_app_info_14
- row counts: {'nid_new_registration_06': 81, 'sem_app_info_14': 81}
- nearest excluded: [{'tag': 'nid_vote_without_card_01', 'min_sim': 0.9754, 'reason': 'non_reciprocal'}, {'tag': 'new_voter_how_to_get_card', 'min_sim': 0.9732, 'reason': 'non_reciprocal'}, {'tag': 'unable_to_answer', 'min_sim': 0.9724, 'reason': 'non_reciprocal'}]

### `fam_7c106922`  min=0.976  avg=0.980
- tags: otp_entry_screen, otp_length, otp_send_button
- row counts: {'otp_entry_screen': 101, 'otp_length': 100, 'otp_send_button': 100}
- nearest excluded: [{'tag': 'otp_not_received', 'min_sim': 0.9823, 'reason': 'covered_by_other_family'}, {'tag': 'process_requirements', 'min_sim': 0.978, 'reason': 'non_reciprocal'}, {'tag': 'otp_delivery_time', 'min_sim': 0.978, 'reason': 'covered_by_other_family'}]

### `fam_d349df22`  min=0.976  avg=0.983
- tags: nid_reissue_04, online_application_pending, reissue_apply_why_pending_and_urgent_reissue_seven_days_over
- row counts: {'nid_reissue_04': 81, 'online_application_pending': 175, 'reissue_apply_why_pending_and_urgent_reissue_seven_days_over': 180}
- nearest excluded: [{'tag': 'nid_reissue_05', 'min_sim': 0.9811, 'reason': 'below_pair_threshold'}, {'tag': 'photo_done_years_ago_but_no_card', 'min_sim': 0.9784, 'reason': 'non_reciprocal'}, {'tag': 'correction_application_status', 'min_sim': 0.9781, 'reason': 'non_reciprocal'}]

### `fam_663bbd3a`  min=0.976  avg=0.976
- tags: birth_date_no_proof_model, old_birth_certificate
- row counts: {'birth_date_no_proof_model': 192, 'old_birth_certificate': 176}
- nearest excluded: [{'tag': 'age_and_dob_correction', 'min_sim': 0.9893, 'reason': 'below_pair_threshold'}, {'tag': 'card_information_correction', 'min_sim': 0.9816, 'reason': 'non_reciprocal'}, {'tag': 'birthplace_correction_new', 'min_sim': 0.9786, 'reason': 'covered_by_other_family'}]

### `fam_b59374ee`  min=0.975  avg=0.980
- tags: after_success_login_button, home_screen_usage, post_ballot_vote_confirmation, process_requirements
- row counts: {'after_success_login_button': 101, 'home_screen_usage': 99, 'post_ballot_vote_confirmation': 101, 'process_requirements': 99}
- nearest excluded: [{'tag': 'app_requirements_internet_gps_vpn', 'min_sim': 0.9871, 'reason': 'covered_by_other_family'}, {'tag': 'login_credentials_required', 'min_sim': 0.9856, 'reason': 'below_pair_threshold'}, {'tag': 'instruction_following_tips', 'min_sim': 0.9828, 'reason': 'covered_by_other_family'}]

### `fam_6485503e`  min=0.975  avg=0.975
- tags: mobile_number_entry, one_mobile_number_how_many_time_registration
- row counts: {'mobile_number_entry': 101, 'one_mobile_number_how_many_time_registration': 171}
- nearest excluded: [{'tag': 'unable_to_answer', 'min_sim': 0.9789, 'reason': 'non_reciprocal'}, {'tag': 'nid_probashi_registration_01', 'min_sim': 0.9752, 'reason': 'covered_by_other_family'}, {'tag': 'cell_phone_number_correct_or_add_document_new', 'min_sim': 0.975, 'reason': 'covered_by_other_family'}]

### `fam_ffc15db9`  min=0.975  avg=0.975
- tags: nid_probashi_biometric_status_03, nid_probashi_biometric_status_04
- row counts: {'nid_probashi_biometric_status_03': 81, 'nid_probashi_biometric_status_04': 80}
- nearest excluded: [{'tag': 'nid_probashi_biometric_status_01', 'min_sim': 0.9809, 'reason': 'below_pair_threshold'}, {'tag': 'nid_probashi_registration_07', 'min_sim': 0.9782, 'reason': 'below_pair_threshold'}, {'tag': 'foreign_resident_action_after_biometrics_new', 'min_sim': 0.9776, 'reason': 'below_pair_threshold'}]

### `fam_60b25d5d`  min=0.975  avg=0.982
- tags: birthplace_correction_new, name_correction_in_nid_card, parents_name_correction_new, spouse_name_correction_new
- row counts: {'birthplace_correction_new': 182, 'name_correction_in_nid_card': 169, 'parents_name_correction_new': 223, 'spouse_name_correction_new': 228}
- nearest excluded: [{'tag': 'birth_certificate_number_correct_or_add_document_new', 'min_sim': 0.9856, 'reason': 'covered_by_other_family'}, {'tag': 'age_and_dob_correction', 'min_sim': 0.9833, 'reason': 'below_pair_threshold'}, {'tag': 'parent_spouse_name_correct_or_add_document_new', 'min_sim': 0.9831, 'reason': 'covered_by_other_family'}]

### `fam_6fe2486a`  min=0.975  avg=0.980
- tags: nid_new_registration_05, nid_new_registration_07, way_to_cancel_correction_application
- row counts: {'nid_new_registration_05': 79, 'nid_new_registration_07': 81, 'way_to_cancel_correction_application': 175}
- nearest excluded: [{'tag': 'wrong_information_in_online_application_what_to_do', 'min_sim': 0.9845, 'reason': 'below_pair_threshold'}, {'tag': 'nid_probashi_registration_02', 'min_sim': 0.9823, 'reason': 'covered_by_other_family'}, {'tag': 'new_voter_application_cancelled', 'min_sim': 0.9822, 'reason': 'covered_by_other_family'}]

### `fam_5d911018`  min=0.975  avg=0.979
- tags: ballot_envelope_postage_free, ballot_envelope_submission, ballot_marking_instructions, ballot_packing_instructions
- row counts: {'ballot_envelope_postage_free': 101, 'ballot_envelope_submission': 101, 'ballot_marking_instructions': 99, 'ballot_packing_instructions': 100}
- nearest excluded: [{'tag': 'multiple_marks_invalid_vote', 'min_sim': 0.9834, 'reason': 'covered_by_other_family'}, {'tag': 'voting_precautions', 'min_sim': 0.9813, 'reason': 'covered_by_other_family'}, {'tag': 'post_ballot_vote_confirmation', 'min_sim': 0.9786, 'reason': 'covered_by_other_family'}]

### `fam_d0baa016`  min=0.975  avg=0.981
- tags: fraction, photo_sms_but_not_in_upzila_other_when, warning_to_prank_caller
- row counts: {'fraction': 574, 'photo_sms_but_not_in_upzila_other_when': 172, 'warning_to_prank_caller': 197}
- nearest excluded: [{'tag': 'request_for_clear_pronunciation', 'min_sim': 0.9913, 'reason': 'covered_by_other_family'}, {'tag': 'goodbye', 'min_sim': 0.9846, 'reason': 'covered_by_other_family'}, {'tag': 'correction_application_status', 'min_sim': 0.9826, 'reason': 'covered_by_other_family'}]

### `fam_b231d74f`  min=0.975  avg=0.980
- tags: login_after_registration, online_portal_registration, registration_start_button, registration_start_country_select
- row counts: {'login_after_registration': 100, 'online_portal_registration': 171, 'registration_start_button': 97, 'registration_start_country_select': 99}
- nearest excluded: [{'tag': 'relogin_credentials_required', 'min_sim': 0.9852, 'reason': 'below_pair_threshold'}, {'tag': 'wish_to_create_nid_smart_card_new', 'min_sim': 0.9849, 'reason': 'below_pair_threshold'}, {'tag': 'login_safety_precautions', 'min_sim': 0.9847, 'reason': 'covered_by_other_family'}]

### `fam_2c75ccf9`  min=0.975  avg=0.978
- tags: address_cahgne_but_no_post_office_or_office_code, nid_probashi_registration_03, no_upzila_in_permananet_address
- row counts: {'address_cahgne_but_no_post_office_or_office_code': 170, 'nid_probashi_registration_03': 81, 'no_upzila_in_permananet_address': 170}
- nearest excluded: [{'tag': 'no_upzila_in_online_registration', 'min_sim': 0.9871, 'reason': 'below_pair_threshold'}, {'tag': 'nid_address_update_03', 'min_sim': 0.9848, 'reason': 'below_pair_threshold'}, {'tag': 'nid_address_update_11', 'min_sim': 0.9828, 'reason': 'covered_by_other_family'}]

### `fam_71aff8dd`  min=0.975  avg=0.977
- tags: no_otp_in_registration, unexpected_online_problem, wrong_password_in_registration
- row counts: {'no_otp_in_registration': 174, 'unexpected_online_problem': 170, 'wrong_password_in_registration': 194}
- nearest excluded: [{'tag': 'wrong_otp_registration', 'min_sim': 0.9823, 'reason': 'covered_by_other_family'}, {'tag': 'photo_done_years_ago_but_no_card', 'min_sim': 0.9823, 'reason': 'non_reciprocal'}, {'tag': 'unable_to_answer', 'min_sim': 0.9811, 'reason': 'non_reciprocal'}]

### `fam_45dee6ce`  min=0.974  avg=0.978
- tags: correction_fee_submitted_but_balance_zero, nid_afis_01, nid_missing_finger_tax_issue_01
- row counts: {'correction_fee_submitted_but_balance_zero': 170, 'nid_afis_01': 80, 'nid_missing_finger_tax_issue_01': 81}
- nearest excluded: [{'tag': 'fingerprint_error_info_not_found', 'min_sim': 0.9931, 'reason': 'covered_by_other_family'}, {'tag': 'nid_biometric_update_not_working_01', 'min_sim': 0.9856, 'reason': 'below_pair_threshold'}, {'tag': 'fingerprint_update_applied_not_approved', 'min_sim': 0.9852, 'reason': 'covered_by_other_family'}]

### `fam_e9198d02`  min=0.974  avg=0.977
- tags: nid_address_update_01, nid_voter_area_transfer_01, nid_voter_area_transfer_06, voter_area_change_online
- row counts: {'nid_address_update_01': 82, 'nid_voter_area_transfer_01': 81, 'nid_voter_area_transfer_06': 80, 'voter_area_change_online': 171}
- nearest excluded: [{'tag': 'nid_address_update_12', 'min_sim': 0.987, 'reason': 'covered_by_other_family'}, {'tag': 'voter_area_change', 'min_sim': 0.9845, 'reason': 'covered_by_other_family'}, {'tag': 'nid_voter_area_transfer_02', 'min_sim': 0.9802, 'reason': 'covered_by_other_family'}]

### `fam_8cebb5f8`  min=0.974  avg=0.974
- tags: nid_smart_card_print_status_01, smart_card_print_problem
- row counts: {'nid_smart_card_print_status_01': 77, 'smart_card_print_problem': 170}
- nearest excluded: [{'tag': 'smart_card_status', 'min_sim': 0.9874, 'reason': 'covered_by_other_family'}, {'tag': 'smart_card_selected_but_get_pdf', 'min_sim': 0.9795, 'reason': 'non_reciprocal'}, {'tag': 'smart_card_ready_sms', 'min_sim': 0.9769, 'reason': 'covered_by_other_family'}]

### `fam_6fb64fba`  min=0.973  avg=0.973
- tags: nid_correction_online_02, nid_probashi_registration_02
- row counts: {'nid_correction_online_02': 81, 'nid_probashi_registration_02': 79}
- nearest excluded: [{'tag': 'wrong_information_in_online_application_what_to_do', 'min_sim': 0.9917, 'reason': 'below_pair_threshold'}, {'tag': 'card_information_correction', 'min_sim': 0.9873, 'reason': 'covered_by_other_family'}, {'tag': 'nid_new_registration_05', 'min_sim': 0.9823, 'reason': 'covered_by_other_family'}]

### `fam_ea245f4d`  min=0.973  avg=0.983
- tags: new_nid_card_age, nid_probashi_registration_04, nid_registration_age, underaged_card_holder_to_voter
- row counts: {'new_nid_card_age': 174, 'nid_probashi_registration_04': 77, 'nid_registration_age': 151, 'underaged_card_holder_to_voter': 168}
- nearest excluded: [{'tag': 'new_voter_pre_condition', 'min_sim': 0.9852, 'reason': 'below_pair_threshold'}, {'tag': 'nid_card_validity_period_01', 'min_sim': 0.9735, 'reason': 'below_pair_threshold'}, {'tag': 'faster_voter_registration', 'min_sim': 0.9734, 'reason': 'non_reciprocal'}]

### `fam_f99688a0`  min=0.973  avg=0.973
- tags: email_optional, nid_required_for_postal_vote
- row counts: {'email_optional': 101, 'nid_required_for_postal_vote': 100}
- nearest excluded: [{'tag': 'accepted_nid_types', 'min_sim': 0.9897, 'reason': 'covered_by_other_family'}, {'tag': 'nid_vote_without_card_01', 'min_sim': 0.9846, 'reason': 'covered_by_other_family'}, {'tag': 'voting_precautions', 'min_sim': 0.9764, 'reason': 'non_reciprocal'}]

### `fam_226226d8`  min=0.973  avg=0.973
- tags: qr_code_clarification, qr_scan_instructions
- row counts: {'qr_code_clarification': 174, 'qr_scan_instructions': 95}
- nearest excluded: [{'tag': 'qr_scan_mandatory', 'min_sim': 0.9861, 'reason': 'below_pair_threshold'}, {'tag': 'qr_scan_button_activation', 'min_sim': 0.9813, 'reason': 'below_pair_threshold'}, {'tag': 'no_otp_in_registration', 'min_sim': 0.9723, 'reason': 'non_reciprocal'}]

### `fam_c3d94fb0`  min=0.972  avg=0.972
- tags: multiple_marks_invalid_vote, one_device_one_voter
- row counts: {'multiple_marks_invalid_vote': 101, 'one_device_one_voter': 101}
- nearest excluded: [{'tag': 'ballot_marking_instructions', 'min_sim': 0.9834, 'reason': 'covered_by_other_family'}, {'tag': 'ballot_packing_instructions', 'min_sim': 0.9801, 'reason': 'covered_by_other_family'}, {'tag': 'voting_precautions', 'min_sim': 0.9773, 'reason': 'non_reciprocal'}]

### `fam_b795619b`  min=0.971  avg=0.975
- tags: nid_misc_03, nid_probashi_application_rejected_02, nid_reissue_07
- row counts: {'nid_misc_03': 81, 'nid_probashi_application_rejected_02': 79, 'nid_reissue_07': 81}
- nearest excluded: [{'tag': 'card_lost_no_info_how_to_get_new', 'min_sim': 0.9912, 'reason': 'below_pair_threshold'}, {'tag': 'voter_slip_lost_how_to_get_nid', 'min_sim': 0.9897, 'reason': 'covered_by_other_family'}, {'tag': 'picture_done_but_lost_or_no_sms_slip', 'min_sim': 0.9847, 'reason': 'covered_by_other_family'}]

### `fam_4ac982a1`  min=0.971  avg=0.978
- tags: accepted_nid_types, new_voter_smart_card_how_to_get, nid_misc_01, nid_vote_without_card_01
- row counts: {'accepted_nid_types': 98, 'new_voter_smart_card_how_to_get': 167, 'nid_misc_01': 81, 'nid_vote_without_card_01': 81}
- nearest excluded: [{'tag': 'new_voter_got_normal_card_when_get_smart_card', 'min_sim': 0.9908, 'reason': 'below_pair_threshold'}, {'tag': 'new_voter_how_to_get_card', 'min_sim': 0.9902, 'reason': 'covered_by_other_family'}, {'tag': 'nid_required_for_postal_vote', 'min_sim': 0.9897, 'reason': 'covered_by_other_family'}]

### `fam_c4737dab`  min=0.971  avg=0.980
- tags: foreign_address_entry_tips, foreign_address_fields, nid_address_update_10, nid_probashi_address_requirement_01
- row counts: {'foreign_address_entry_tips': 100, 'foreign_address_fields': 97, 'nid_address_update_10': 80, 'nid_probashi_address_requirement_01': 81}
- nearest excluded: [{'tag': 'foreign_resident_card_registration_process', 'min_sim': 0.9807, 'reason': 'non_reciprocal'}, {'tag': 'foreign_resident_multiple_registration_new', 'min_sim': 0.977, 'reason': 'covered_by_other_family'}, {'tag': 'unable_to_answer', 'min_sim': 0.9767, 'reason': 'non_reciprocal'}]

### `fam_f59c12eb`  min=0.970  avg=0.979
- tags: nid_opportunity_for_foreign_spouse, nid_probashi_eligibility_01, nid_probashi_registration_01
- row counts: {'nid_opportunity_for_foreign_spouse': 167, 'nid_probashi_eligibility_01': 81, 'nid_probashi_registration_01': 81}
- nearest excluded: [{'tag': 'nid_dual_citizenship_02', 'min_sim': 0.9803, 'reason': 'below_pair_threshold'}, {'tag': 'mobile_number_entry', 'min_sim': 0.9752, 'reason': 'covered_by_other_family'}, {'tag': 'nid_probashi_registration_07', 'min_sim': 0.9739, 'reason': 'non_reciprocal'}]

### `fam_0b2c0c8e`  min=0.970  avg=0.983
- tags: new_voter_apply_online_what_documents_needed, new_voter_online_what_documents_to_submit, new_voter_registered_where_To_submit_the_downloaded_form, nid_probashi_registration_05
- row counts: {'new_voter_apply_online_what_documents_needed': 170, 'new_voter_online_what_documents_to_submit': 163, 'new_voter_registered_where_To_submit_the_downloaded_form': 190, 'nid_probashi_registration_05': 77}
- nearest excluded: [{'tag': 'unable_to_answer', 'min_sim': 0.9798, 'reason': 'non_reciprocal'}, {'tag': 'reissue_supporting_document', 'min_sim': 0.9778, 'reason': 'covered_by_other_family'}, {'tag': 'post_submission_verification', 'min_sim': 0.9765, 'reason': 'non_reciprocal'}]

### `fam_e6ba98fa`  min=0.970  avg=0.975
- tags: bkash_mfs_payment_process, correction_fee_payment, nid_fee_payment_channels_01, rocket_mfs_payment_process
- row counts: {'bkash_mfs_payment_process': 182, 'correction_fee_payment': 165, 'nid_fee_payment_channels_01': 81, 'rocket_mfs_payment_process': 172}
- nearest excluded: [{'tag': 'instant_card_correction_fee', 'min_sim': 0.9826, 'reason': 'below_pair_threshold'}, {'tag': 'card_correction_fees', 'min_sim': 0.9818, 'reason': 'below_pair_threshold'}, {'tag': 'how_to_apply_for_correction_in_online_portal', 'min_sim': 0.9804, 'reason': 'covered_by_other_family'}]

### `fam_75a76930`  min=0.969  avg=0.969
- tags: nid_fee_03, nid_probashi_registration_06
- row counts: {'nid_fee_03': 81, 'nid_probashi_registration_06': 81}
- nearest excluded: [{'tag': 'foreign_resident_action_after_biometrics_new', 'min_sim': 0.979, 'reason': 'below_pair_threshold'}, {'tag': 'nid_probashi_biometric_status_03', 'min_sim': 0.9702, 'reason': 'non_reciprocal'}, {'tag': 'nid_biometric_update_status_01', 'min_sim': 0.9683, 'reason': 'non_reciprocal'}]

### `fam_274604dd`  min=0.969  avg=0.977
- tags: liveness_check_avoid, liveness_check_purpose, liveness_check_steps, post_qr_liveness_requirement
- row counts: {'liveness_check_avoid': 101, 'liveness_check_purpose': 101, 'liveness_check_steps': 100, 'post_qr_liveness_requirement': 101}
- nearest excluded: [{'tag': 'instruction_following_tips', 'min_sim': 0.9806, 'reason': 'non_reciprocal'}, {'tag': 'after_success_login_button', 'min_sim': 0.9792, 'reason': 'covered_by_other_family'}, {'tag': 'voting_precautions', 'min_sim': 0.9778, 'reason': 'non_reciprocal'}]

### `fam_b9373afe`  min=0.967  avg=0.971
- tags: nid_address_update_07, nid_address_update_11, wrong_address_consequence
- row counts: {'nid_address_update_07': 81, 'nid_address_update_11': 81, 'wrong_address_consequence': 101}
- nearest excluded: [{'tag': 'address_cahgne_but_no_post_office_or_office_code', 'min_sim': 0.9828, 'reason': 'covered_by_other_family'}, {'tag': 'incorrect_info_restart', 'min_sim': 0.9787, 'reason': 'non_reciprocal'}, {'tag': 'wrong_mobile_number_consequence', 'min_sim': 0.9786, 'reason': 'covered_by_other_family'}]

### `fam_8709afb5`  min=0.966  avg=0.975
- tags: dual_citizenship_certificate_of_which_countires_not_required_for_nid, ministry_provide_dual_citizenship_certificate, whether_nid_card_required_for_dual_citizenship, whom_to_communicate_for_dual_citizenship_certificate
- row counts: {'dual_citizenship_certificate_of_which_countires_not_required_for_nid': 170, 'ministry_provide_dual_citizenship_certificate': 171, 'whether_nid_card_required_for_dual_citizenship': 171, 'whom_to_communicate_for_dual_citizenship_certificate': 171}
- nearest excluded: [{'tag': 'nid_dual_citizenship_01', 'min_sim': 0.9818, 'reason': 'below_pair_threshold'}, {'tag': 'unable_to_answer', 'min_sim': 0.9665, 'reason': 'non_reciprocal'}, {'tag': 'double_voter_registration_issue', 'min_sim': 0.9657, 'reason': 'non_reciprocal'}]

### `fam_fc4827e2`  min=0.966  avg=0.978
- tags: nid_address_update_09, nid_misc_07, nid_new_registration_02, parent_no_card_failed_to_be_voter
- row counts: {'nid_address_update_09': 81, 'nid_misc_07': 81, 'nid_new_registration_02': 81, 'parent_no_card_failed_to_be_voter': 171}
- nearest excluded: [{'tag': 'nid_vote_without_card_01', 'min_sim': 0.9802, 'reason': 'covered_by_other_family'}, {'tag': 'father_and_my_adress_different', 'min_sim': 0.9791, 'reason': 'below_pair_threshold'}, {'tag': 'no_own_house_in_voter_area', 'min_sim': 0.9779, 'reason': 'non_reciprocal'}]

### `fam_deb97b2b`  min=0.965  avg=0.975
- tags: nid_information_misentry_correction_procedure, nid_photo_guidelines, photo_correction
- row counts: {'nid_information_misentry_correction_procedure': 74, 'nid_photo_guidelines': 101, 'photo_correction': 182}
- nearest excluded: [{'tag': 'card_information_correction', 'min_sim': 0.9881, 'reason': 'covered_by_other_family'}, {'tag': 'card_damaged_how_to_get_new_card', 'min_sim': 0.9831, 'reason': 'non_reciprocal'}, {'tag': 'age_and_dob_correction', 'min_sim': 0.9825, 'reason': 'below_pair_threshold'}]

### `fam_7f9f5089`  min=0.962  avg=0.975
- tags: how_many_wife_name_can_be_added_in_nid_card, husband_name_previous_card, no_husband_name_in_smart_card
- row counts: {'how_many_wife_name_can_be_added_in_nid_card': 174, 'husband_name_previous_card': 168, 'no_husband_name_in_smart_card': 169}
- nearest excluded: [{'tag': 'spouse_name_correction_new', 'min_sim': 0.9742, 'reason': 'non_reciprocal'}, {'tag': 'husband_talak_name_remove', 'min_sim': 0.9735, 'reason': 'below_pair_threshold'}, {'tag': 'voter_slip_lost_how_to_get_new_card', 'min_sim': 0.965, 'reason': 'non_reciprocal'}]

### `fam_a4e176eb`  min=0.960  avg=0.977
- tags: foreign_resident_passport_expaired_card_registration_process, nid_required_for_registration, passport_info_required, passport_number_optional
- row counts: {'foreign_resident_passport_expaired_card_registration_process': 175, 'nid_required_for_registration': 102, 'passport_info_required': 100, 'passport_number_optional': 101}
- nearest excluded: [{'tag': 'foreign_resident_card_registration_process', 'min_sim': 0.9824, 'reason': 'non_reciprocal'}, {'tag': 'login_credentials_required', 'min_sim': 0.9804, 'reason': 'below_pair_threshold'}, {'tag': 'nid_dual_citizenship_02', 'min_sim': 0.9801, 'reason': 'below_pair_threshold'}]

### `fam_744a4c6a`  min=0.960  avg=0.971
- tags: all_documents_correct_but_application_rejected_why, foreign_resident_card_registration_rejected, new_voter_but_card_rejected, nid_probashi_application_rejected_01
- row counts: {'all_documents_correct_but_application_rejected_why': 172, 'foreign_resident_card_registration_rejected': 172, 'new_voter_but_card_rejected': 173, 'nid_probashi_application_rejected_01': 79}
- nearest excluded: [{'tag': 'unable_to_answer', 'min_sim': 0.9794, 'reason': 'non_reciprocal'}, {'tag': 'nid_new_registration_04', 'min_sim': 0.9792, 'reason': 'non_reciprocal'}, {'tag': 'foreign_resident_card_picture_done__inquery_done_no_msg_new', 'min_sim': 0.9786, 'reason': 'covered_by_other_family'}]

### `fam_445186bc`  min=0.953  avg=0.965
- tags: nid_probashi_approval_pending_01, nid_probashi_approval_pending_02, nid_reissue_approvers_01
- row counts: {'nid_probashi_approval_pending_01': 81, 'nid_probashi_approval_pending_02': 81, 'nid_reissue_approvers_01': 81}
- nearest excluded: [{'tag': 'nid_reissue_09', 'min_sim': 0.9806, 'reason': 'below_pair_threshold'}, {'tag': 'foreign_resident_card_picture_done__inquery_done_no_msg_new', 'min_sim': 0.9778, 'reason': 'covered_by_other_family'}, {'tag': 'unable_to_answer', 'min_sim': 0.977, 'reason': 'non_reciprocal'}]

### `fam_8a827234`  min=0.949  avg=0.967
- tags: birth_certificate_and_ssc_certificate_different_information, reigister_as_new_voter_but_birth_registration_different_address, ssc_certificate_for_correction
- row counts: {'birth_certificate_and_ssc_certificate_different_information': 172, 'reigister_as_new_voter_but_birth_registration_different_address': 171, 'ssc_certificate_for_correction': 171}
- nearest excluded: [{'tag': 'voter_area_change', 'min_sim': 0.9855, 'reason': 'covered_by_other_family'}, {'tag': 'nid_address_update_12', 'min_sim': 0.9813, 'reason': 'covered_by_other_family'}, {'tag': 'father_and_my_adress_different', 'min_sim': 0.9808, 'reason': 'below_pair_threshold'}]

