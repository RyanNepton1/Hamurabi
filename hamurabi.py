#!/usr/bin/env python3
"""Hamurabi - a text-based resource management game.

Based on the classic "Hamurabi" (The Sumer Game) by Doug Dyment and the
BASIC adaptation popularized by David H. Ahl.

You rule the ancient kingdom of Sumeria for 10 years. Each year you decide how
many acres to buy or sell, how much grain to feed your people, and how many
acres to plant with seed. Manage wisely to keep your people alive and grow the
kingdom.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GameState:
    year: int = 1
    population: int = 100
    bushels: int = 2800
    acres: int = 1000

    # Results from the previous year (reported at start of current year)
    starved: int = 0
    immigrants: int = 5
    plague: bool = False
    harvest_bushels: int = 3000
    yield_per_acre: int = 3
    rats_ate: int = 200

    # Cumulative tracking
    total_starved: int = 0
    years_played: int = 0
    cumulative_starvation_pct: float = 0.0

    # End state
    game_over: bool = False
    impeached: bool = False
    end_message: str = ""


class HamurabiGame:
    """Core Hamurabi game logic.

    Classic-style rules:
    - 10-year term.
    - Land trades for 17-26 bushels per acre (random each year).
    - Each person needs 20 bushels of grain per year to live.
    - Planting requires 1 bushel of seed for every 2 acres.
    - Each person can farm at most 10 acres.
    - Harvest yields 1-5 bushels per acre (random).
    - Rats may eat part of stored grain in some years.
    - A plague may kill half the population (~15% chance per year).
    - Immigration depends on prosperity/resources.
    - If more than 45% of the population starves in one year, the ruler is
      impeached and the game ends immediately.
    """

    MAX_YEARS = 10
    BUSHELS_PER_PERSON = 20
    ACRES_PER_PERSON = 10

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()
        self.state = GameState()

    def current_land_price(self) -> int:
        """Return the trading price (bushels per acre) for the current year."""
        return self.rng.randint(17, 26)

    @staticmethod
    def seed_for_acres(acres: int) -> int:
        """Bushels of seed required to plant the given acres."""
        return (acres + 1) // 2  # 1 bushel plants 2 acres

    def validate_turn(
        self,
        acres_bought: int,
        acres_sold: int,
        bushels_fed: int,
        acres_planted: int,
        land_price: int,
    ) -> Tuple[bool, str]:
        """Validate a full set of yearly decisions against current state.

        Returns (ok, error_message). Decisions are validated in order as they
        affect available resources: buy/sell, feed, plant.
        """
        s = self.state
        bushels = s.bushels
        acres = s.acres
        population = s.population

        # Trade
        if acres_bought < 0 or acres_sold < 0 or bushels_fed < 0 or acres_planted < 0:
            return False, "Hamurabi: I cannot do what you wish. Negative amounts are not allowed."
        if acres_bought > 0 and acres_sold > 0:
            return False, "Hamurabi: You cannot buy and sell land in the same year."

        if acres_bought > 0:
            cost = acres_bought * land_price
            if cost > bushels:
                return (
                    False,
                    f"Hamurabi: Think again. You have only {bushels} bushels of grain. "
                    f"Now then,",
                )
            bushels -= cost
            acres += acres_bought
        elif acres_sold > 0:
            if acres_sold > acres:
                return (
                    False,
                    f"Hamurabi: Think again. You own only {acres} acres. Now then,",
                )
            bushels += acres_sold * land_price
            acres -= acres_sold

        # Feed
        if bushels_fed > bushels:
            return (
                False,
                f"Hamurabi: Think again. You have only {bushels} bushels of grain. Now then,",
            )
        bushels -= bushels_fed

        # Plant
        if acres_planted > acres:
            return (
                False,
                f"Hamurabi: Think again. You own only {acres} acres. Now then,",
            )
        seed = self.seed_for_acres(acres_planted)
        if seed > bushels:
            return (
                False,
                f"Hamurabi: Think again. You have only {bushels} bushels of grain left for seed. "
                "Now then,",
            )
        max_farmable = population * self.ACRES_PER_PERSON
        if acres_planted > max_farmable:
            return (
                False,
                f"Hamurabi: Think again. You have only {population} people to tend the fields, "
                f"and each can farm at most {self.ACRES_PER_PERSON} acres. Now then,",
            )

        return True, ""

    def advance_year(
        self,
        acres_bought: int,
        acres_sold: int,
        bushels_fed: int,
        acres_planted: int,
        land_price: int,
    ) -> GameState:
        """Apply one year's decisions and advance to the next year.

        Callers should validate decisions first via `validate_turn`.
        """
        s = self.state
        if s.game_over:
            return s

        # 1. Trade
        if acres_bought > 0:
            s.bushels -= acres_bought * land_price
            s.acres += acres_bought
        elif acres_sold > 0:
            s.bushels += acres_sold * land_price
            s.acres -= acres_sold

        # 2. Feed people
        s.bushels -= bushels_fed
        fed_people = bushels_fed // self.BUSHELS_PER_PERSON
        starved = max(0, s.population - fed_people)
        s.starved = starved
        s.total_starved += starved

        # Impeachment check: more than 45% starved in a single year
        if s.population > 0 and starved > s.population * 0.45:
            s.impeached = True
            s.game_over = True
            s.population = max(0, s.population - starved)
            s.years_played += 1
            s.cumulative_starvation_pct += (starved / max(1, s.population + starved)) * 100
            s.end_message = (
                f"You starved {starved} people in one year!!!\n"
                "Due to this extreme mismanagement you have not only\n"
                "been impeached and thrown out of office but you have\n"
                "also been declared a national fink!!!!"
            )
            return s

        pop_before_losses = s.population
        starvation_pct = (starved / pop_before_losses * 100) if pop_before_losses else 0.0

        # 3. Plant and harvest
        seed = self.seed_for_acres(acres_planted)
        s.bushels -= seed
        s.yield_per_acre = self.rng.randint(1, 5)
        s.harvest_bushels = acres_planted * s.yield_per_acre
        s.bushels += s.harvest_bushels

        # 4. Rats
        # Classic behavior: in some years rats eat a fraction of stored grain.
        if self.rng.randint(0, 1) == 0:
            # rats eat between ~1/5 and ~1/2 in bad cases in simplified form
            # use a random divisor to vary severity
            fraction = self.rng.uniform(0.1, 0.3)
            s.rats_ate = int(s.bushels * fraction)
        else:
            s.rats_ate = 0
        s.bushels -= s.rats_ate
        if s.bushels < 0:
            s.bushels = 0

        # 5. Immigration
        survivors = s.population - starved
        if survivors <= 0:
            s.immigrants = 0
            s.population = 0
            s.years_played += 1
            s.cumulative_starvation_pct += starvation_pct
            s.game_over = True
            s.end_message = "You have no people left! The kingdom has perished."
            return s

        # Common Hamurabi immigration formula
        s.immigrants = int(
            (20 * s.acres + s.bushels) / (100 * survivors) + 1
        )
        s.immigrants = max(0, min(s.immigrants, 5000))

        s.population = survivors + s.immigrants

        # 6. Plague (~15% chance)
        s.plague = self.rng.randint(1, 100) <= 15
        if s.plague:
            s.population = s.population // 2

        s.years_played += 1
        s.cumulative_starvation_pct += starvation_pct

        if s.population <= 0:
            s.population = 0
            s.game_over = True
            s.end_message = "Your entire population has perished!"
            return s

        # 7. Advance year or end term
        s.year += 1
        if s.year > self.MAX_YEARS:
            s.game_over = True
            s.end_message = self._final_assessment()

        return s

    def _final_assessment(self) -> str:
        s = self.state
        acres_per_person = s.acres / max(1, s.population)
        avg_starvation_pct = s.cumulative_starvation_pct / max(1, s.years_played)

        lines = []
        lines.append(
            f"In your {s.years_played}-year term of office, {avg_starvation_pct:.1f} percent of the"
        )
        lines.append(
            f"population starved per year on average, for a total of {s.total_starved} deaths."
        )
        lines.append(
            f"You started with 10 acres per person and ended with {acres_per_person:.0f} acres per person."
        )
        lines.append("")

        if avg_starvation_pct > 33 or acres_per_person < 7:
            lines.append("Due to this extreme mismanagement you have not only")
            lines.append("been impeached and thrown out of office but you have")
            lines.append("also been declared a national fink!!!!")
        elif avg_starvation_pct > 10 or acres_per_person < 9:
            lines.append("Your heavy-handed performance smacks of Nero and Ivan IV.")
            lines.append("The people (remaining) find you an unpleasant ruler, and,")
            lines.append("frankly, hate your guts!!")
        elif avg_starvation_pct > 3 or acres_per_person < 10:
            assassinated = int(self.rng.random() * max(1, s.population) * 0.8)
            lines.append("Your performance could have been somewhat better, but")
            lines.append(f"really wasn't too bad at all. {assassinated} people")
            lines.append("would dearly like to see you assassinated but we all have our")
            lines.append("trivial problems.")
        else:
            lines.append("A fantastic performance!!! Charlemagne, Disraeli, and")
            lines.append("Jefferson combined could not have done better!")

        return "\n".join(lines)

    # --- Convenience runner for programmatic use ---
    def run_year(
        self,
        acres_bought: int,
        acres_sold: int,
        bushels_fed: int,
        acres_planted: int,
        land_price: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """Validate and apply a year. Returns (ok, error_or_empty)."""
        if self.state.game_over:
            return False, "The game is already over."
        if land_price is None:
            land_price = self.current_land_price()
        ok, err = self.validate_turn(
            acres_bought, acres_sold, bushels_fed, acres_planted, land_price
        )
        if not ok:
            return False, err
        self.advance_year(acres_bought, acres_sold, bushels_fed, acres_planted, land_price)
        return True, ""


def _print_report(game: HamurabiGame, land_price: int) -> None:
    s = game.state
    print()
    print("Hamurabi: I beg to report to you,")
    print(f"in year {s.year}, {s.starved} people starved, {s.immigrants} came to the city.")
    if s.plague:
        print("A horrible plague struck! Half the people died.")
    print(f"Population is now {s.population}.")
    print(f"The city now owns {s.acres} acres.")
    print(f"You harvested {s.yield_per_acre} bushels per acre.")
    print(f"Rats ate {s.rats_ate} bushels.")
    print(f"You now have {s.bushels} bushels in store.")
    print(f"Land is trading at {land_price} bushels per acre.")


def _ask_int(prompt: str) -> Optional[int]:
    """Prompt for an integer. Returns None on EOF."""
    while True:
        try:
            value = input(prompt)
        except EOFError:
            print()
            return None
        value = value.strip()
        if value == "":
            print("Hamurabi: Please enter a number.")
            continue
        try:
            # Accept integers; reject floats-like strings with decimals
            return int(value)
        except ValueError:
            print("Hamurabi: Please enter a whole number.")


def _prompt_turn(game: HamurabiGame, land_price: int):
    """Interactively prompt the player for one year's decisions.

    Returns (acres_bought, acres_sold, bushels_fed, acres_planted) or None
    if input is exhausted.
    """
    s = game.state
    while True:
        # Buy / sell
        acres_bought = 0
        acres_sold = 0
        ans = _ask_int("How many acres do you wish to buy? ")
        if ans is None:
            return None
        if ans > 0:
            acres_bought = ans
        else:
            acres_bought = 0
            ans = _ask_int("How many acres do you wish to sell? ")
            if ans is None:
                return None
            acres_sold = max(0, ans)

        ans = _ask_int("How many bushels do you wish to feed your people? ")
        if ans is None:
            return None
        bushels_fed = ans

        ans = _ask_int("How many acres do you wish to plant with seed? ")
        if ans is None:
            return None
        acres_planted = ans

        ok, err = game.validate_turn(
            acres_bought, acres_sold, bushels_fed, acres_planted, land_price
        )
        if ok:
            return acres_bought, acres_sold, bushels_fed, acres_planted
        print(err)


def play_interactive(seed: Optional[int] = None) -> GameState:
    """Run an interactive game of Hamurabi in the terminal."""
    rng = random.Random(seed)
    game = HamurabiGame(rng=rng)

    print("Try your hand at governing ancient Sumeria")
    print("for a ten-year term of office.")
    print()
    print("Rules:")
    print(f"- Each person needs {HamurabiGame.BUSHELS_PER_PERSON} bushels of grain per year to live.")
    print("- It takes 1 bushel of seed to plant 2 acres.")
    print(f"- Each person can farm at most {HamurabiGame.ACRES_PER_PERSON} acres.")
    print("- Land trades for 17 to 26 bushels per acre each year.")
    print("- If more than 45% of your people starve in one year, you are impeached.")
    print()

    while not game.state.game_over:
        land_price = game.current_land_price()
        _print_report(game, land_price)
        decisions = _prompt_turn(game, land_price)
        if decisions is None:
            print("\nHamurabi: So long for now.")
            return game.state
        acres_bought, acres_sold, bushels_fed, acres_planted = decisions
        game.advance_year(acres_bought, acres_sold, bushels_fed, acres_planted, land_price)

        if game.state.impeached or (game.state.game_over and game.state.population == 0):
            print()
            print(game.state.end_message)
            return game.state

    # Normal end of term
    print()
    print(game.state.end_message)
    return game.state


def main() -> None:
    play_interactive()


if __name__ == "__main__":
    main()
