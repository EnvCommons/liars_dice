import asyncio
import textarena as ta
import re
import openai
from typing import List, Tuple
from pydantic import BaseModel
from openreward.environments import Environment, JSONObject, ToolOutput, TextBlock, tool


class TaskSpec(BaseModel):
    id: str
    env_id: str
    seed: int
    variant: str = ""


class BidParams(BaseModel, extra="forbid"):
    quantity: int
    face: int


class EmptyParams(BaseModel, extra="forbid"):
    pass


class LiarsDiceEnvironment(Environment):
    GAME_NAME = "LiarsDice"
    VARIANTS = [
        "LiarsDice-v0-small",
        "LiarsDice-v0-small-train",
        "LiarsDice-v0-small-raw",
        "LiarsDice-v0",
        "LiarsDice-v0-train",
        "LiarsDice-v0-raw",
        "LiarsDice-v0-large",
        "LiarsDice-v0-large-train",
        "LiarsDice-v0-large-raw",
    ]
    NUM_TASKS_PER_VARIANT = 50
    AGENT_PLAYER_ID = 0
    NUM_PLAYERS = 2

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)
        self.secrets = secrets

        api_key = secrets.get("openai_api_key")
        if not api_key:
            raise ValueError("openai_api_key required in secrets for LiarsDice (LLM opponent)")
        self.opponent_client = openai.AsyncClient(api_key=api_key)

        self.ta_env = ta.make(env_id=self.config.env_id)
        self.game_done = False
        self.turn_count = 0

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        tasks = []
        for variant_id in cls.VARIANTS:
            for seed_idx in range(cls.NUM_TASKS_PER_VARIANT):
                seed = seed_idx if split == "train" else seed_idx + 10000
                tasks.append({
                    "id": f"{variant_id}_seed{seed}",
                    "env_id": variant_id,
                    "seed": seed,
                    "variant": variant_id,
                })
        return tasks

    def _format_observation(self, observation) -> str:
        if isinstance(observation, str):
            match = None
            for m in re.finditer(r'^\[(?!GAME\])[^\]]+\].*$', observation, re.MULTILINE):
                match = m
            if match:
                return observation[match.end():].lstrip('\n')
            return observation
        if isinstance(observation, list):
            if not observation:
                return ""
            last = observation[-1]
            if isinstance(last, tuple) and len(last) >= 2:
                return str(last[1])
            return str(last)
        return str(observation)

    def _map_reward(self, ta_rewards: dict, player_id: int) -> float:
        raw = ta_rewards.get(player_id, 0)
        return max(0.0, min(1.0, (raw + 1.0) / 2.0))

    async def _get_opponent_action(self, observation: str, player_id: int) -> str:
        system_prompt = (
            f"You are playing Liar's Dice as Player {player_id}. "
            f"Respond with ONLY [Bid: X, Y] or [Call]. No explanations."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": observation},
        ]
        try:
            response = await self.opponent_client.chat.completions.create(
                model="gpt-5-mini",
                messages=messages,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "[Call]"

    async def _run_opponent_turns(self, current_player_id: int, current_observation) -> str:
        while current_player_id != self.AGENT_PLAYER_ID:
            obs_text = self._format_observation(current_observation)
            opponent_action = await self._get_opponent_action(obs_text, current_player_id)
            done, info = self.ta_env.step(action=opponent_action)
            if done:
                self.game_done = True
                return ""
            current_player_id, current_observation = self.ta_env.get_observation()
        return self._format_observation(current_observation)

    async def get_prompt(self) -> List[TextBlock]:
        self.ta_env.reset(num_players=self.NUM_PLAYERS, seed=self.config.seed)
        player_id, observation = self.ta_env.get_observation()

        if player_id != self.AGENT_PLAYER_ID:
            obs_text = await self._run_opponent_turns(player_id, observation)
        else:
            obs_text = self._format_observation(observation)

        prompt = (
            f"You are playing Liar's Dice, a bluffing dice game.\n\n"
            f"Rules:\n"
            f"- Each player has dice hidden from others\n"
            f"- On your turn, bid a quantity and face value, or call the last bid a bluff\n"
            f"- Bids must increase in quantity or face value\n"
            f"- If you call: if actual count < bid, bidder loses a die; otherwise caller loses a die\n"
            f"- Last player with dice wins\n\n"
            f"Use the 'bid' tool with quantity and face, or 'call_bluff' to challenge.\n\n"
            f"{obs_text}"
        )
        return [TextBlock(text=prompt)]

    def _handle_game_end(self) -> Tuple[str, float, bool]:
        rewards, game_info = self.ta_env.close()
        reward = self._map_reward(rewards, self.AGENT_PLAYER_ID)
        reason = ""
        if isinstance(game_info, dict) and self.AGENT_PLAYER_ID in game_info:
            reason = game_info[self.AGENT_PLAYER_ID].get("reason", "")
        summary = f"Game Over! Your reward: {reward:.2f}"
        if reason:
            summary += f"\n{reason}"
        self.game_done = True
        return summary, reward, True

    async def _do_action(self, action: str) -> ToolOutput:
        if self.game_done:
            return ToolOutput(
                blocks=[TextBlock(text="Game is already over.")],
                metadata={"error": "game_finished"},
                reward=0.0,
                finished=True,
            )

        done, info = self.ta_env.step(action=action)
        self.turn_count += 1

        if done:
            summary, reward, finished = self._handle_game_end()
            return ToolOutput(
                blocks=[TextBlock(text=summary)],
                metadata={"turn": self.turn_count, "reward": reward},
                reward=reward,
                finished=True,
            )

        player_id, observation = self.ta_env.get_observation()
        if player_id != self.AGENT_PLAYER_ID:
            obs_text = await self._run_opponent_turns(player_id, observation)
            if self.game_done:
                summary, reward, finished = self._handle_game_end()
                return ToolOutput(
                    blocks=[TextBlock(text=summary)],
                    metadata={"turn": self.turn_count, "reward": reward},
                    reward=reward,
                    finished=True,
                )
        else:
            obs_text = self._format_observation(observation)

        return ToolOutput(
            blocks=[TextBlock(text=obs_text)],
            metadata={"turn": self.turn_count},
            reward=0.0,
            finished=False,
        )

    @tool
    async def bid(self, params: BidParams) -> ToolOutput:
        """Place a bid. Specify quantity (number of dice) and face value (1-6)."""
        return await self._do_action(f"[Bid: {params.quantity}, {params.face}]")

    @tool
    async def call_bluff(self, params: EmptyParams) -> ToolOutput:
        """Call the opponent's last bid as a bluff."""
        return await self._do_action("[Call]")
