"""
Expert allocation helper based on the paper-style affinity algorithm.

This module is not the main RSMA-MoE scheduler. It keeps a standalone expert
placement routine that can be used for experiments with paper variables such as
EAfE, EAfR, and the expert-to-server placement matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import random
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class ExpertAllocationResult:
    """Result returned by the expert allocation routine."""

    p_matrix_ep: List[List[int]]  # Expert-to-server placement matrix, shape N x E.
    p_matrix_req: List[List[float]]  # Request-to-server soft assignment, shape K x E.
    expert_loads: List[float]
    server_loads: List[float]
    total_affinity_score: float


class ExpertAllocator:
    """Paper-style expert allocator using affinity and local search.

    Symbols:
    - N: number of experts.
    - E: number of edge servers.
    - K: number of requests/subtasks.
    - Cp: K x N matrix where Cp[k][n] is the demand of request k for expert n.
    - alpha_e: weight for expert-expert affinity, EAfE.
    - beta_e: weight for request-expert affinity, EAfR.
    - gamma_e: load-balancing penalty weight.
    """

    def __init__(
        self,
        n_experts: int,
        n_servers: int,
        n_requests: int,
        experts: Dict[str, Dict],
        servers: List[Dict],
        requests: List[Dict],
        alpha_e: float = 0.33,
        beta_e: float = 0.33,
        gamma_e: float = 0.34,
        ft_steps: int = 100,
    ):
        self.N = n_experts
        self.E = n_servers
        self.K = n_requests
        self.experts = experts
        self.servers = servers
        self.requests = requests

        self.alpha_e = alpha_e
        self.beta_e = beta_e
        self.gamma_e = gamma_e
        self.ft_steps = ft_steps

        self.expert_ids = list(experts.keys())
        self.server_ids = [server["server_id"] for server in servers]
        self.request_ids = [request["id"] for request in requests]

    def allocate(self, Cp: List[List[float]]) -> ExpertAllocationResult:
        """Allocate experts to edge servers using the affinity score."""
        logger.info("Start expert allocation: N=%s, E=%s, K=%s", self.N, self.E, self.K)

        expert_loads = self._compute_expert_loads(Cp)
        logger.debug("Expert loads: %s", expert_loads)

        sorted_expert_indices = sorted(
            range(self.N),
            key=lambda index: expert_loads[index],
            reverse=True,
        )

        p_matrix_ep = [[0] * self.E for _ in range(self.N)]
        p_matrix_req = [[1 / self.E] * self.E for _ in range(self.K)]

        server_available = [1] * self.E
        expert_count_by_server = [0] * self.E
        load_by_server = [0.0] * self.E

        for expert_idx in sorted_expert_indices:
            expert_aff_scores = []

            for server_idx in range(self.E):
                eafe = self._compute_expert_expert_affinity(
                    server_available,
                    p_matrix_ep,
                    Cp,
                    server_idx,
                )
                eafr = self._compute_request_expert_affinity(
                    server_available,
                    p_matrix_req,
                    Cp,
                    expert_idx,
                    server_idx,
                )
                load_penalty = self.gamma_e * load_by_server[server_idx]
                affinity_score = (self.alpha_e * eafe) + (self.beta_e * eafr) - load_penalty
                expert_aff_scores.append(affinity_score)

            best_server = max(range(self.E), key=lambda index: expert_aff_scores[index])
            p_matrix_ep[expert_idx][best_server] = 1
            expert_count_by_server[best_server] += 1
            load_by_server[best_server] += expert_loads[expert_idx]

            logger.debug("Expert %s -> Server %s", self.expert_ids[expert_idx], best_server)

            if expert_count_by_server[best_server] >= self.N / self.E:
                server_available[best_server] = 0

        p_matrix_ep = self._local_search_swap(
            p_matrix_ep,
            Cp,
            expert_loads,
            load_by_server,
        )
        server_loads = self._compute_server_loads(p_matrix_ep, expert_loads)
        total_score = self._compute_total_affinity_score(p_matrix_ep, Cp, expert_loads)

        result = ExpertAllocationResult(
            p_matrix_ep=p_matrix_ep,
            p_matrix_req=p_matrix_req,
            expert_loads=expert_loads,
            server_loads=server_loads,
            total_affinity_score=total_score,
        )

        logger.info("Expert allocation complete. Total affinity score: %.2f", total_score)
        return result

    def _compute_expert_loads(self, Cp: List[List[float]]) -> List[float]:
        """Compute expert load: loads[n] = sum_k Cp[k][n]."""
        loads = [0.0] * self.N
        for request_idx in range(self.K):
            for expert_idx in range(self.N):
                loads[expert_idx] += Cp[request_idx][expert_idx]
        return loads

    def _compute_expert_expert_affinity(
        self,
        server_available: List[int],
        p_matrix_ep: List[List[int]],
        Cp: List[List[float]],
        server_idx: int,
    ) -> float:
        """Compute EAfE for the candidate server."""
        if server_available[server_idx] == 0:
            return float("-inf")

        experts_in_server = [
            expert_idx
            for expert_idx in range(self.N)
            if p_matrix_ep[expert_idx][server_idx] == 1
        ]
        if not experts_in_server:
            return 0.0

        affinity = 0.0
        for existing_expert in experts_in_server:
            common_requests = 0
            for request_idx in range(self.K):
                if Cp[request_idx][existing_expert] > 0 and Cp[request_idx][server_idx] > 0:
                    common_requests += 1
            affinity += common_requests / max(self.K, 1)

        return affinity / max(len(experts_in_server), 1)

    def _compute_request_expert_affinity(
        self,
        server_available: List[int],
        p_matrix_req: List[List[float]],
        Cp: List[List[float]],
        expert_idx: int,
        server_idx: int,
    ) -> float:
        """Compute EAfR for a candidate expert and server."""
        if server_available[server_idx] == 0:
            return float("-inf")

        affinity = 0.0
        for request_idx in range(self.K):
            request_server_affinity = p_matrix_req[request_idx][server_idx]
            expert_requirement = Cp[request_idx][expert_idx]
            affinity += request_server_affinity * expert_requirement

        return affinity / max(self.K, 1)

    def _local_search_swap(
        self,
        p_matrix_ep: List[List[int]],
        Cp: List[List[float]],
        expert_loads: List[float],
        load_by_server: List[float],
    ) -> List[List[int]]:
        """Improve placement by randomly swapping experts between servers."""
        del load_by_server
        logger.debug("Start local search for %s iterations", self.ft_steps)

        best_p_matrix = [row[:] for row in p_matrix_ep]
        best_score = self._compute_total_affinity_score(best_p_matrix, Cp, expert_loads)

        for _ in range(self.ft_steps):
            server_1, server_2 = random.sample(range(self.E), 2)

            experts_server_1 = [
                expert_idx
                for expert_idx in range(self.N)
                if p_matrix_ep[expert_idx][server_1] == 1
            ]
            experts_server_2 = [
                expert_idx
                for expert_idx in range(self.N)
                if p_matrix_ep[expert_idx][server_2] == 1
            ]

            if not experts_server_1 or not experts_server_2:
                continue

            expert_1 = random.choice(experts_server_1)
            expert_2 = random.choice(experts_server_2)

            affinity_gain = self._compute_affinity_gain(
                p_matrix_ep,
                Cp,
                expert_loads,
                expert_1,
                expert_2,
                server_1,
                server_2,
            )

            if affinity_gain > 0:
                self._swap_experts(p_matrix_ep, expert_1, expert_2, server_1, server_2)
                logger.debug(
                    "Swap experts %s and %s; affinity gain: %.2f",
                    expert_1,
                    expert_2,
                    affinity_gain,
                )

                current_score = self._compute_total_affinity_score(
                    p_matrix_ep,
                    Cp,
                    expert_loads,
                )
                if current_score > best_score:
                    best_p_matrix = [row[:] for row in p_matrix_ep]
                    best_score = current_score

        return best_p_matrix

    def _compute_affinity_gain(
        self,
        p_matrix_ep: List[List[int]],
        Cp: List[List[float]],
        expert_loads: List[float],
        expert_1: int,
        expert_2: int,
        server_1: int,
        server_2: int,
    ) -> float:
        """Return the affinity gain from swapping two experts."""
        original_score = self._compute_total_affinity_score(p_matrix_ep, Cp, expert_loads)

        self._swap_experts(p_matrix_ep, expert_1, expert_2, server_1, server_2)
        new_score = self._compute_total_affinity_score(p_matrix_ep, Cp, expert_loads)
        self._swap_experts(p_matrix_ep, expert_1, expert_2, server_1, server_2)

        return new_score - original_score

    def _swap_experts(
        self,
        p_matrix_ep: List[List[int]],
        expert_1: int,
        expert_2: int,
        server_1: int,
        server_2: int,
    ) -> None:
        """Swap two experts between two servers."""
        p_matrix_ep[expert_1][server_1] = 0
        p_matrix_ep[expert_1][server_2] = 1
        p_matrix_ep[expert_2][server_2] = 0
        p_matrix_ep[expert_2][server_1] = 1

    def _compute_server_loads(
        self,
        p_matrix_ep: List[List[int]],
        expert_loads: List[float],
    ) -> List[float]:
        """Compute total expert load assigned to each server."""
        server_loads = [0.0] * self.E

        for expert_idx in range(self.N):
            for server_idx in range(self.E):
                if p_matrix_ep[expert_idx][server_idx] == 1:
                    server_loads[server_idx] += expert_loads[expert_idx]
                    break

        return server_loads

    def _compute_total_affinity_score(
        self,
        p_matrix_ep: List[List[int]],
        Cp: List[List[float]],
        expert_loads: List[float],
    ) -> float:
        """Compute total expert-expert affinity across all servers."""
        del expert_loads
        score = 0.0

        for server_idx in range(self.E):
            experts_in_server = [
                expert_idx
                for expert_idx in range(self.N)
                if p_matrix_ep[expert_idx][server_idx] == 1
            ]
            if not experts_in_server:
                continue

            for index, expert_1 in enumerate(experts_in_server):
                for expert_2 in experts_in_server[index + 1:]:
                    common = sum(
                        1
                        for request_idx in range(self.K)
                        if Cp[request_idx][expert_1] > 0 and Cp[request_idx][expert_2] > 0
                    )
                    score += common

        return score

    def get_allocation_summary(self, result: ExpertAllocationResult) -> str:
        """Return a readable allocation summary."""
        summary = "\n=== Expert Allocation Summary ===\n"
        summary += f"Total affinity score: {result.total_affinity_score:.2f}\n\n"

        summary += "Server load distribution:\n"
        for server_idx, load in enumerate(result.server_loads):
            expert_count = sum(row[server_idx] for row in result.p_matrix_ep)
            summary += f"  Server {self.server_ids[server_idx]}: {load:.2f} (experts: {expert_count})\n"

        summary += "\nExpert placement:\n"
        for expert_idx, expert_id in enumerate(self.expert_ids):
            for server_idx in range(self.E):
                if result.p_matrix_ep[expert_idx][server_idx] == 1:
                    summary += f"  {expert_id} -> {self.server_ids[server_idx]}\n"
                    break

        return summary
