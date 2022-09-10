import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType


def get_report(data):
    attack_input = AttackInputData(
        logits_train=data.logits_train,
        logits_test=data.logits_test,
        loss_train=data.loss_train,
        loss_test=data.loss_test,
        labels_train=data.train_labels,
        labels_test=data.test_labels
    )
    slicing_spec = SlicingSpec(
        entire_dataset=True,
        by_class=True,
        by_percentiles=False,
        by_classification_correctness=True
    )

    attack_types = [
        # AttackType.THRESHOLD_ATTACK,
        AttackType.RANDOM_FOREST,
        AttackType.K_NEAREST_NEIGHBORS,
        # AttackType.THRESHOLD_ENTROPY_ATTACK
    ]

    attacks_result = mia.run_attacks(attack_input=attack_input,
                                     slicing_spec=slicing_spec,
                                     attack_types=attack_types)

    return attacks_result, attack_input, slicing_spec


def get_advantage(attacks_result):
    max_auc_attacker = attacks_result.get_result_with_max_auc()
    max_advantage_attacker = attacks_result.get_result_with_max_attacker_advantage()
    return max_auc_attacker, max_advantage_attacker


def get_prob(attack_input, slicing_spec):
    membership_probability_results = mia.run_membership_probability_analysis(attack_input,
                                                                             slicing_spec)
    return membership_probability_results
