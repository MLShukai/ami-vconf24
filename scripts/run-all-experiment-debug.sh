#!/bin/bash

# Function to handle errors and send error message
handle_error() {
    local exit_code=$?
    local command=$1
    if [ $exit_code -ne 0 ]; then
        echo "Error occurred. Failed command:"
        echo "$command"
        echo "Exit code: $exit_code"
        exit $exit_code
    fi
}

# Define array of experiment configurations
declare -a experiments=(
    # Random observation action log
    "experiment=random_observation_action_log \
    max_uptime=3"

    # Bool mask i-jepa with videos (large)
    "experiment=bool_mask_i_jepa_with_videos \
    models=bool_mask_i_jepa_large \
    max_uptime=10"

    # Bool mask i-jepa with videos (small)
    "experiment=bool_mask_i_jepa_with_videos \
    models=bool_mask_i_jepa_small \
    max_uptime=10"

    # Learn only sioconv (large)
    "experiment=learn_only_sioconv \
    models=learn_only_sioconv_large \
    max_uptime=10"

    # Learn only sioconv (small)
    "experiment=learn_only_sioconv \
    models=learn_only_sioconv_small \
    max_uptime=10"

    # Learn only sioconv with random permutation sampler
    "experiment=learn_only_sioconv \
    models=learn_only_sioconv_large \
    max_uptime=10 \
    trainers.forward_dynamics.partial_sampler._target_=ami.trainers.components.random_permutation_sampler.RandomPermutationSampler"

    # Learn i-jepa sioconv
    "experiment=learn_i_jepa_sioconv \
    max_uptime=10"

    # I-jepa sioconv ppo multi step (SimpleWorld, small)
    "experiment=i_jepa_sioconv_ppo_multi_step \
    max_uptime=10 \
    interaction/environment=unity \
    interaction.environment.file_path=/workspace/unity_executables/SimpleWorld/SimpleWorld.x86_64 \
    models=i_jepa_sioconv_resnetpolicy_small \
    models.policy_value.model.policy_head.action_choices_per_category.path=ami.interactions.environments.actuators.vrchat_osc_discrete_actuator.ACTION_CHOICES_PER_CATEGORY_MASKED_JUMP_RUN"

    # I-jepa sioconv ppo multi step (SimpleWorld, large)
    "experiment=i_jepa_sioconv_ppo_multi_step \
    max_uptime=10 \
    interaction/environment=unity \
    interaction.environment.file_path=/workspace/unity_executables/SimpleWorld/SimpleWorld.x86_64 \
    models=i_jepa_sioconv_resnetpolicy_large \
    models.policy_value.model.policy_head.action_choices_per_category.path=ami.interactions.environments.actuators.vrchat_osc_discrete_actuator.ACTION_CHOICES_PER_CATEGORY_MASKED_JUMP_RUN"

    # I-jepa sioconv ppo multi step (NoisyWorld, small)
    "experiment=i_jepa_sioconv_ppo_multi_step \
    max_uptime=10 \
    interaction/environment=unity \
    interaction.environment.file_path=/workspace/unity_executables/NoisyWorld2023/NoisyWorld2023.x86_64 \
    models=i_jepa_sioconv_resnetpolicy_small \
    models.policy_value.model.policy_head.action_choices_per_category.path=ami.interactions.environments.actuators.vrchat_osc_discrete_actuator.ACTION_CHOICES_PER_CATEGORY_MASKED_JUMP_RUN"

    # I-jepa sioconv ppo multi step (NoisyWorld, large)
    "experiment=i_jepa_sioconv_ppo_multi_step \
    max_uptime=10 \
    interaction/environment=unity \
    interaction.environment.file_path=/workspace/unity_executables/NoisyWorld2023/NoisyWorld2023.x86_64 \
    models=i_jepa_sioconv_resnetpolicy_large \
    models.policy_value.model.policy_head.action_choices_per_category.path=ami.interactions.environments.actuators.vrchat_osc_discrete_actuator.ACTION_CHOICES_PER_CATEGORY_MASKED_JUMP_RUN"
)

# Execute each experiment
for experiment in "${experiments[@]}"; do
    cmd="python scripts/launch.py ${experiment}"
    echo "Executing command:"
    echo "$cmd"
    echo "-------------------"
    $cmd
    handle_error "$cmd"
done
