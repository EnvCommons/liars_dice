# LiarsDice

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/LiarsDice)

## Description

**LiarsDice** is an environment for evaluating agents on strategic bluffing and probabilistic reasoning in Liar's Dice, a dice game where players must bid or call bluffs. This environment wraps the LiarsDice implementation from [TextArena](https://github.com/LeonGuertler/TextArena), a framework for text-based game environments.

## Capabilities

- Testing probabilistic reasoning under uncertainty
- Evaluating strategic bluffing and deception detection
- Assessing risk management in bid escalation
- Testing opponent modeling and behavioral adaptation

## Compute Requirements

LiarsDice does not require a sandbox. It has minimal compute requirements.

## License

[MIT](https://github.com/LeonGuertler/TextArena/blob/main/LICENSE).

## Tasks

There are two splits: train (450 tasks) and test (450 tasks). Each split contains 50 tasks across each of 9 variants:

- **LiarsDice-v0-small**
- **LiarsDice-v0-small-train**
- **LiarsDice-v0-small-raw**
- **LiarsDice-v0**
- **LiarsDice-v0-train**
- **LiarsDice-v0-raw**
- **LiarsDice-v0-large**
- **LiarsDice-v0-large-train**
- **LiarsDice-v0-large-raw**

Each task is seeded for reproducibility.

## Reward Structure

This is a sparse reward environment. Rewards are mapped from TextArena's native range of {-1, 0, 1} to {0.0, 0.5, 1.0} via `(raw + 1) / 2`.

We do not use LLM graders for this environment; reward is determined programmatically.

## Data

Game state is generated procedurally by the TextArena engine using seeded randomness. No external data files are required.

## Tools

Agents are given two tools:

- `bid(quantity, face)`: Place a bid. Specify quantity (number of dice) and face value (1-6).
- `call_bluff()`: Call the opponent's last bid as a bluff.

## Time Horizon

LiarsDice is a multi-turn environment.

## Environment Difficulty

Medium to Hard. Liar's Dice requires probabilistic calculation, strategic bluffing, and reading opponent patterns. Success depends on balancing aggressive bidding with timely bluff calls, adapting to partial information.

## Other Environment Requirements

This environment requires an OpenAI API key (passed via secrets) to power the LLM opponent.

## Safety

Agents in LiarsDice interact only with a dice game and have no access to external systems, the internet, or sensitive data. The environment does not present safety risks.

## Citations

```bibtex
@software{textarena2024,
  author    = {Guertler, Leon and Banting, Wilfried and Pignatelli, Eduardo},
  title     = {TextArena},
  year      = {2024},
  publisher = {GitHub},
  url       = {https://github.com/LeonGuertler/TextArena}
}
```
