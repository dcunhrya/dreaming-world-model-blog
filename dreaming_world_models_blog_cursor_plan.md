# Cursor Plan: Blog on Dreaming, Train-Time Compute, and World Models in RL

## Working title

**Learning to Dream: Train-Time Compute with World Models**

## Goal

Create a blog post in the same style as the existing posts:

- `Learning to Scale GPU Workloads with Reinforcement Learning`
- `An Interactive Guide to GPU Utilization`

The post should be technical but approachable. It should explain “dreaming” in reinforcement learning as **train-time compute spent inside a learned world model**, then show the idea with a small runnable exercise.

The blog should have a practical systems/research flavor:

1. Start with a real bottleneck.
2. Define the algorithmic problem clearly.
3. Build a small simulator.
4. Compare baselines.
5. Show plots.
6. Explain where the idea works.
7. Explain where it breaks.
8. Connect the toy example to modern research and healthcare.

## Style target

Match the prior blog style:

- Clear section headings.
- Short explanatory paragraphs.
- Technical but not overly formal.
- Use tables for state/action/reward definitions.
- Use small equations when they clarify the mental model.
- Include runnable code snippets.
- Include interactive sliders or plots if possible.
- End with practical takeaways and caveats.
- Avoid hype unless immediately paired with a failure mode.

The post should feel like:

> “Here is the simple version. Here is the simulator. Here is what happens when we change the knob. Here is why the research frontier is harder.”

## Core thesis

**Dreaming is train-time compute spent generating imagined experience from a learned model of the environment.**

In standard RL, an agent learns from real interaction:

```text
observe state -> choose action -> receive next state and reward -> update policy
```

In model-based RL, the agent also learns a world model:

```text
state + action -> predicted next state + predicted reward
```

Once the world model is good enough, the agent can train on imagined rollouts:

```text
real experience -> train world model -> imagine rollouts -> update policy -> act again
```

This is the key distinction:

| Concept | Plain-English meaning |
|---|---|
| Model-free RL | Learn behavior directly from real or logged experience |
| Model-based RL | Learn a model of consequences, then use it for planning or training |
| World model | A learned simulator of action-conditioned future states and rewards |
| Dreaming | Rolling the learned model forward to create imagined experience |
| Train-time compute | Extra compute spent before deployment to improve the policy |
| Test-time compute | Extra compute spent while acting, such as search or planning |

The memorable line:

> A world model is not necessarily a video generator. It is a consequence model. It only needs to preserve the parts of the future that matter for action.

## Opening draft

Use this as the first-pass introduction:

```markdown
Modern RL has a simple problem: real experience is expensive.

A robot that drops a cup, a web agent that edits the wrong field, or a clinical agent that orders the wrong test cannot learn purely by trial and error in the real world. The classical way around this is simulation. But hand-built simulators are brittle, expensive, and domain-specific.

World models offer a different path. Instead of writing the simulator by hand, the agent learns one from data. It watches how the environment changes after actions, compresses those dynamics into a predictive model, and then uses that model to generate imagined rollouts. Those rollouts are the agent's dreams.

In this post, we will build the smallest useful version of that idea. We will start with Q-learning in a GridWorld, add a learned tabular world model, and then spend extra train-time compute on imagined updates. The result is Dyna-Q: an old idea that feels surprisingly close to the current world-model frontier.

The point is not that dreams are magic. The point is that dreams are cheap experience. When the world model is accurate, they can make learning dramatically more sample efficient. When the world model is wrong, they can teach the agent a policy that only works in the dream.
```

## Section outline

### 1. Introduction

Motivation:

- Real environment interaction is expensive.
- Model-free RL can be sample-inefficient.
- Simulators help, but hand-built simulators are hard to write.
- A learned world model is a simulator learned from data.
- Dreaming is training inside that simulator.

Key line:

```text
Test-time compute asks: can the model think longer before acting?
Train-time dreaming asks: can the model cheaply generate more experience before deployment?
```

### 2. The algorithmic problem

Define the MDP:

```text
s_t: state
a_t: action
r_t: reward
P(s_{t+1} | s_t, a_t): transition dynamics
π(a_t | s_t): policy
```

Model-free RL learns a policy or value function directly from experience:

```text
Q(s_t, a_t) <- Q(s_t, a_t) + α [r_t + γ max_a Q(s_{t+1}, a) - Q(s_t, a_t)]
```

Model-based RL also learns:

```text
P_hat(s_{t+1} | s_t, a_t)
R_hat(s_t, a_t)
```

Then the agent can perform updates on either real transitions or imagined transitions.

### 3. What is a world model?

Use this section to prevent confusion.

A world model can mean several related things:

| Type | What it predicts | Example |
|---|---|---|
| Tabular dynamics model | Next state and reward from observed counts | Dyna-Q |
| Neural latent dynamics model | Future latent state and reward | Dreamer |
| Planning model | Reward, value, and policy-relevant latent dynamics | MuZero |
| Interactive generative world | Action-conditioned future observations | Genie, Minecraft world models |

The important idea:

```text
A world model does not have to perfectly reconstruct the world.
It has to preserve enough structure for the agent to choose better actions.
```

This is where to introduce **value equivalence**: a model can be useful if it preserves the quantities that matter for planning, even if it ignores irrelevant details.

### 4. What is dreaming?

Dreaming is just imagined rollout:

```text
start from known state s
choose action a
sample predicted next state s' from P_hat
sample predicted reward r from R_hat
update the policy or value function
repeat
```

Diagram:

```text
real env step
    ↓
update world model
    ↓
sample imagined transitions
    ↓
update Q / policy / critic
    ↓
act again
```

Keep this concrete:

```text
No dreaming: 1 real transition = 1 learning update
Dreaming with k=20: 1 real transition = 1 real update + 20 imagined updates
```

That is the train-time compute knob.

### 5. Toy exercise: Dyna-Q GridWorld

Build a small GridWorld.

Environment:

| Component | Definition |
|---|---|
| Grid | 8x8 |
| Start | lower-left |
| Goal | upper-right |
| Walls | fixed blocked cells |
| State | agent row and column |
| Actions | up, down, left, right |
| Reward | +1 for goal, -0.01 per step, -0.2 for wall collision |

World model:

```python
transition_counts[(s, a)][s_next] += 1
reward_sum[(s, a)] += r
reward_count[(s, a)] += 1
```

Dream step:

```python
s, a = sample_previously_seen_state_action()
s_next = sample_from_transition_model(s, a)
r = average_reward_model(s, a)
q_update(s, a, r, s_next)
```

Run conditions:

| Agent | Description |
|---|---|
| Q-learning | No dreaming |
| Dyna-Q k=5 | 5 imagined updates per real step |
| Dyna-Q k=20 | 20 imagined updates per real step |
| Dyna-Q k=100 | Heavy train-time compute |
| Noisy Dyna-Q | World model has transition noise |

### 6. Plots

Add these plots:

1. **Average return vs real environment steps**
   - Shows sample efficiency.
   - Dreaming should learn faster in terms of real interaction.

2. **Average return vs total updates**
   - Shows compute efficiency.
   - Heavy dreaming may look less impressive once compute is counted.

3. **Steps to goal vs episodes**
   - Intuitive and easy to read.

4. **Model error vs final real return**
   - Shows the failure mode.

5. **Interactive dreaming budget slider**
   - Slider: `dream_updates_per_real_step = 0, 1, 5, 20, 100`
   - Show how learning improves with more simulated updates, until compute or model error dominates.

6. **Interactive model noise slider**
   - Slider: `model_noise = 0.0, 0.05, 0.1, 0.2`
   - Show that bad dreams can degrade real performance.

### 7. The key result

Expected narrative:

```markdown
With no dreaming, the agent only learns from real transitions. It eventually solves the grid, but it wastes many real environment steps rediscovering nearby consequences.

With Dyna-Q, each real transition also updates a world model. After every real step, the agent samples old state-action pairs from memory and performs imagined Q-learning updates. These imagined updates propagate value information faster. The goal reward spreads backward through the grid before the agent physically revisits every path.

This is the sample-efficiency gain from train-time compute. We are not making the environment easier. We are spending extra computation to reuse experience more aggressively.
```

### 8. When dreaming fails

This is the most important credibility section.

Use a transition-noise experiment.

Possible explanation:

```markdown
The danger is that the agent can overfit to its dream.

If the learned transition model predicts that moving right from a wall sometimes teleports the agent closer to the goal, Q-learning has no way to know that the transition is fake during imagined updates. The value function will assign credit to actions that only work inside the model.

This is model exploitation. The agent does not become irrational. It becomes rational with respect to the wrong world.
```

Takeaway:

```text
More dreaming helps only while model error is low enough.
```

This should be one of the main plots.

### 9. The promise of dreaming

Add this section after the toy experiment.

Main argument:

```markdown
The promise of dreaming is not just that RL agents can solve GridWorld faster. The promise is that agents may eventually generate useful training environments for themselves in domains where real data is scarce, expensive, unsafe, or slow to collect.
```

Examples:

#### Example A: Web and software agents

A coding or web agent can dream up synthetic tasks:

```text
- broken unit tests
- fake GitHub issues
- corrupted config files
- API migration tasks
- multi-step browser workflows
```

The agent can then practice:

```text
observe repo -> inspect files -> edit code -> run tests -> receive reward
```

Why this matters:

- Many real tasks are private and cannot be shared.
- Environments can generate infinite variants.
- Rewards can be programmatic: tests pass, type checks pass, lints pass, hidden eval succeeds.

#### Example B: Robotics

A robot can dream physical outcomes:

```text
grasp object -> object slips
push cup -> cup rotates
open drawer -> object appears
```

The agent can train on imagined interactions before touching the real world.

But the model must preserve action-conditioned physics. A visually pretty video model is not enough if contact dynamics are wrong.

#### Example C: Healthcare

This is where the connection is strongest, but be careful.

Healthcare has all the reasons dreaming is attractive:

| Constraint | Why dreaming helps |
|---|---|
| Scarce labels | Generate more training tasks from limited expert-labeled cases |
| Unsafe exploration | Practice in simulated EHR or patient environments before deployment |
| Long-horizon workflows | Train agents across multi-step chart review, ordering, follow-up, documentation |
| Hidden state | Force agents to decide what information to acquire next |
| Expensive feedback | Use clinician-authored rubrics or structured gold labels as rewards |

Possible healthcare environment examples:

```text
1. Simulated EHR navigation
   Agent receives a task: "Does this patient need anticoagulation adjustment?"
   Actions: search labs, inspect notes, open medication list, check contraindications.
   Reward: correct final answer, minimal unnecessary chart access, guideline-concordant reasoning.

2. Active multimodal acquisition
   Agent starts with incomplete patient information.
   Actions: request image, lab, history question, prior report, vitals trend.
   Reward: correct diagnosis or triage decision minus acquisition cost.

3. Synthetic rare-case curriculum
   A model generates variants of rare clinical cases:
   - atypical presentations
   - missing data
   - noisy history
   - conflicting notes
   - comorbidities
   The agent practices robust information gathering before being evaluated on real held-out cases.

4. ICU decision sandbox
   Agent acts in a patient-state simulator.
   Actions: change ventilator settings, order labs, adjust fluids, escalate care.
   Reward: clinically defined outcome proxy plus safety penalties.
```

Important caveat:

```markdown
In healthcare, dreaming should not mean letting a model hallucinate patients and then trusting the resulting policy. It should mean building constrained, auditable training environments where synthetic rollouts are checked against clinical knowledge, real data distributions, and held-out evaluation.
```

Best healthcare thesis:

```text
Healthcare does not need unconstrained dream worlds.
It needs verified clinical sandboxes.
```

Possible bridge sentence:

```markdown
This is the healthcare version of the GridWorld failure mode. If the world model invents a shortcut, the agent will learn it. In a game, that produces a funny policy. In medicine, it produces unsafe behavior. So the central research problem is not simply generating more environments. It is validating which imagined transitions are safe enough to learn from.
```

### 10. Research progression

Use this short historical arc:

```markdown
The idea is old. Dyna mixed real experience with hypothetical experience from a learned model. World Models showed that neural agents could learn compact latent simulations from pixels. PlaNet and Dreamer moved policy optimization into latent imagination. MuZero showed that a model does not need to predict pixels if it predicts the quantities needed for planning. DreamerV3 and Dreamer4 push the recipe toward broader domains and harder sparse-reward tasks. Genie and V-JEPA show the broader foundation-model version: large models that learn predictive structure from video and interaction.
```

### 11. References to include

Use these references in the blog.

#### Classical model-based RL

1. **Sutton, 1990 — Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming**
   - Dyna-Q origin.
   - Key idea: mix real trial-and-error with hypothetical trial-and-error from a world model.
   - Link: https://www.sciencedirect.com/science/chapter/edited-volume/pii/B9781558601413500304

2. **Sutton et al., 2012 — Dyna-Style Planning with Linear Function Approximation and Prioritized Sweeping**
   - Useful for “imaginary experience from the world model.”
   - Link: https://arxiv.org/abs/1206.3285

#### Neural world models and imagination

3. **Ha and Schmidhuber, 2018 — World Models**
   - Popularized the term in deep RL.
   - Trained an agent partly or entirely inside its hallucinated dream.
   - Link: https://arxiv.org/abs/1803.10122

4. **Hafner et al., 2019 — Dream to Control: Learning Behaviors by Latent Imagination**
   - Dreamer.
   - Learns long-horizon behavior from images by latent imagination.
   - Link: https://arxiv.org/abs/1912.01603

5. **Hafner et al., 2025 — Mastering Diverse Control Tasks through World Models**
   - DreamerV3 Nature paper.
   - General world-model RL across diverse domains.
   - Link: https://www.nature.com/articles/s41586-025-08744-2

6. **Hafner et al., 2025 — Training Agents Inside of Scalable World Models**
   - Dreamer 4.
   - Offline-data world model used for imagination training in Minecraft.
   - Link: https://arxiv.org/abs/2509.24527

#### Planning with learned models

7. **Schrittwieser et al., 2020 — Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model**
   - MuZero.
   - Important because the model predicts planning-relevant quantities rather than reconstructing the full environment.
   - Link: https://www.nature.com/articles/s41586-020-03051-4

8. **Grimm et al., 2020 — The Value Equivalence Principle for Model-Based Reinforcement Learning**
   - A model should capture what matters for value-based planning.
   - Link: https://arxiv.org/abs/2011.03506

#### LLMs, generated environments, and world models

9. **EnvGen, 2024 — Generating and Adapting Environments via LLMs for Training Embodied Agents**
   - LLMs generate environment configurations for training smaller RL agents.
   - Link: https://arxiv.org/abs/2403.12014

10. **Scaling Environments for LLM Agents in the Era of Learning from Interaction, 2025**
   - Survey framing environment scaling for LLM agents.
   - Link: https://arxiv.org/abs/2511.09586

11. **Google DeepMind Genie 3, 2025**
   - Real-time interactive world model that generates environments from text prompts.
   - Link: https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/

12. **Meta V-JEPA 2, 2025**
   - Self-supervised video world model for understanding, prediction, planning, and robot control.
   - Link: https://ai.meta.com/research/vjepa/

#### Healthcare agent environments

13. **MedAgentBench, 2025 — A Realistic Virtual EHR Environment to Benchmark Medical LLM Agents**
   - Interactive FHIR-style EHR environment for medical agents.
   - Link: https://arxiv.org/abs/2501.14654

14. **AgentClinic — A multimodal benchmark for tool-using clinical AI agents**
   - Simulated clinical environments with patient interaction, multimodal data collection, and tools.
   - Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC13324710/

15. **PhysicianBench, 2026 — Evaluating LLM Agents in Real-World EHR Environments**
   - Long-horizon physician tasks in EHR environments.
   - Link: https://arxiv.org/abs/2605.02240

16. **HealthAgentBench, 2026 — Realistic Agentic Healthcare Environments**
   - Suite of healthcare agent tasks across patient-journey workflows.
   - Link: https://arxiv.org/abs/2606.31179

17. **ICU ventilator RL environment, 2026**
   - Patient simulator grounded in MIMIC-IV and eICU for ICU ventilator settings.
   - Link: https://ojs.aaai.org/index.php/AAAI/article/view/39081

## Code implementation plan

### Files

Create a small repo-style blog demo:

```text
dreaming-world-models/
  README.md
  pyproject.toml
  src/
    env.py
    agents.py
    dyna.py
    experiments.py
    plotting.py
  notebooks/
    dreaming_gridworld.ipynb
  website/
    components/
      DreamingBudgetSlider.tsx
      ModelNoiseSlider.tsx
    data/
      dyna_results.json
```

### `env.py`

Implement:

```python
class GridWorld:
    def __init__(
        self,
        size=8,
        walls=None,
        start=(7, 0),
        goal=(0, 7),
        step_penalty=-0.01,
        wall_penalty=-0.2,
        goal_reward=1.0,
        slip_prob=0.0,
        seed=0,
    ):
        ...
```

API:

```python
obs = env.reset()
next_obs, reward, terminated, truncated, info = env.step(action)
```

### `agents.py`

Implement Q-learning:

```python
def q_update(Q, s, a, r, s_next, alpha, gamma):
    best_next = np.max(Q[s_next])
    td_target = r + gamma * best_next
    Q[s][a] += alpha * (td_target - Q[s][a])
```

### `dyna.py`

Implement tabular world model:

```python
class TabularWorldModel:
    def update(self, s, a, r, s_next):
        ...

    def sample(self):
        ...
```

Implement Dyna-Q:

```python
for episode in range(num_episodes):
    s = env.reset()

    while not done:
        a = epsilon_greedy(Q, s)
        s_next, r, done, _, _ = env.step(a)

        q_update(Q, s, a, r, s_next)
        model.update(s, a, r, s_next)

        for _ in range(dream_updates_per_step):
            s_hat, a_hat, r_hat, s_next_hat = model.sample()
            q_update(Q, s_hat, a_hat, r_hat, s_next_hat)

        s = s_next
```

### Model-noise experiment

Add:

```python
class NoisyWorldModel(TabularWorldModel):
    def sample(self):
        s, a, r, s_next = super().sample()
        if rng.random() < model_noise:
            s_next = random_valid_state()
        return s, a, r, s_next
```

Run:

```python
dream_updates_per_step = [0, 1, 5, 20, 100]
model_noise = [0.0, 0.05, 0.10, 0.20]
seeds = range(20)
```

Metrics:

```text
episode_return
steps_to_goal
real_env_steps
imagined_updates
total_updates
success_rate
```

### Plotting

Generate JSON for website:

```json
{
  "curves": [
    {
      "agent": "Dyna-Q k=20",
      "seed": 0,
      "episode": 12,
      "real_env_steps": 842,
      "total_updates": 17682,
      "return": 0.74,
      "steps_to_goal": 31
    }
  ]
}
```

Plots:

- `return_vs_real_steps`
- `return_vs_total_updates`
- `steps_to_goal_vs_episode`
- `final_return_vs_model_noise`
- `sample_efficiency_table`

## Suggested blog conclusion

```markdown
Dreaming gives RL a different scaling axis.

Instead of only collecting more real experience, the agent can spend more compute replaying and recombining what it already knows. In a tabular GridWorld, that looks like Dyna-Q sampling imagined transitions. In Dreamer, it looks like actor-critic learning inside latent trajectories. In emerging foundation world models, it may look like agents training inside generated environments.

The caveat is that the dream has to be good enough. A wrong model can produce a policy that exploits hallucinated transitions. The agent is not confused; it is optimizing exactly what we gave it.

That is why the most interesting version of dreaming is not unlimited synthetic data. It is verified synthetic interaction. This is especially true in healthcare, where data is scarce, exploration is unsafe, and the cost of a false shortcut is high.

The frontier is not simply bigger policies. It is better environments: learned, constrained, auditable worlds where agents can practice before they act.
```

## Cursor prompt

Copy this into Cursor:

```text
Build a blog post titled "Learning to Dream: Train-Time Compute with World Models" in the same structure and tone as my prior technical blogs on RL autoscaling and GPU utilization.

The post should explain dreaming in reinforcement learning as train-time compute spent inside a learned world model. It should include a runnable Dyna-Q GridWorld exercise comparing Q-learning with Dyna-Q at different dreaming budgets.

Use this narrative:
1. Real environment interaction is expensive.
2. A world model is a learned consequence model: state + action -> next state + reward.
3. Dreaming means rolling the model forward to create imagined transitions.
4. Dreaming is a train-time compute knob: more imagined updates per real step.
5. Dyna-Q is the simplest concrete version.
6. More dreaming improves sample efficiency when the model is accurate.
7. More dreaming can hurt when the model is wrong.
8. The modern frontier is agents training inside richer learned worlds.
9. The promise for LLMs is generated environments for scarce-data problems.
10. Healthcare is a compelling but safety-critical case: simulated EHRs, active multimodal acquisition, rare-case curricula, and ICU decision sandboxes.
11. The healthcare caveat is that synthetic clinical environments must be constrained, auditable, and validated.

Implement:
- Python GridWorld environment.
- Q-learning baseline.
- Dyna-Q agent with dream_updates_per_real_step.
- Noisy world model ablation.
- Experiments over k = [0, 1, 5, 20, 100] and model_noise = [0, 0.05, 0.10, 0.20].
- Plots for return vs real environment steps, return vs total updates, steps to goal, and final return vs model noise.
- Optional React/MDX sliders for dreaming budget and model noise.

Use equations:
- Q-learning TD update.
- Learned transition model P_hat(s' | s, a).
- Learned reward model R_hat(s, a).
- Latent world model z_{t+1} ~ pθ(z_{t+1} | z_t, a_t).

Include references:
- Sutton 1990 Dyna
- Sutton et al. 2012 Dyna-style planning
- Ha and Schmidhuber 2018 World Models
- Hafner et al. 2019 Dreamer
- Hafner et al. 2025 DreamerV3
- Hafner et al. 2025 Dreamer4
- Schrittwieser et al. 2020 MuZero
- Grimm et al. 2020 Value Equivalence
- EnvGen 2024
- Scaling Environments for LLM Agents 2025
- Genie 3
- V-JEPA 2
- MedAgentBench
- AgentClinic
- PhysicianBench
- HealthAgentBench
- ICU ventilator RL environment

Write in a practical, direct style. Avoid hype. Every optimistic claim should have a caveat.
```
