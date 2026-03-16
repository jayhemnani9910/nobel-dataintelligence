"""
SIFTS UniProt-to-PDB Mapper

Queries the PDBe SIFTS API to map UniProt accession IDs to
experimentally resolved PDB structures, with local JSON caching.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

logger = logging.getLogger(__name__)

SIFTS_API_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{uid}"


class SIFTSMapper:
    """
    Map UniProt IDs to PDB entries via the PDBe SIFTS REST API.

    Results are cached as JSON files in *cache_dir* to avoid
    redundant network requests across runs.
    """

    def __init__(self, cache_dir: str = "./data/sifts_cache"):
        """
        Initialize mapper.

        Args:
            cache_dir: Directory for cached API responses.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_mapping(self, uniprot_id: str) -> Dict:
        """
        Fetch SIFTS mapping for a single UniProt ID (with caching).

        Args:
            uniprot_id: UniProt accession.

        Returns:
            Parsed JSON response dict, or empty dict on error.
        """
        cache_file = self.cache_dir / f"{uniprot_id}.json"

        if cache_file.exists():
            with open(cache_file, "r") as fh:
                return json.load(fh)

        url = SIFTS_API_URL.format(uid=uniprot_id)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.warning(f"SIFTS request failed for {uniprot_id}: {exc}")
            return {}

        with open(cache_file, "w") as fh:
            json.dump(data, fh)

        return data

    def map_uniprot_to_pdb(self, uniprot_ids: List[str]) -> pd.DataFrame:
        """
        Map a list of UniProt IDs to PDB entries.

        Args:
            uniprot_ids: List of UniProt accession strings.

        Returns:
            DataFrame with columns: uniprot_id, pdb_id, chain,
            resolution, coverage.
        """
        rows: List[Dict] = []

        for uid in uniprot_ids:
            data = self._fetch_mapping(uid)
            if not data:
                continue

            # SIFTS response structure: {uniprot_id: {PDB: {pdb_id: [{...}]}}}
            pdb_mappings = data.get(uid, {}).get("PDB", {})
            for pdb_id, chains in pdb_mappings.items():
                for entry in chains:
                    rows.append({
                        "uniprot_id": uid,
                        "pdb_id": pdb_id,
                        "chain": entry.get("chain_id", ""),
                        "resolution": entry.get("resolution", float("inf")),
                        "coverage": entry.get("coverage", 0.0),
                    })

        df = pd.DataFrame(rows, columns=["uniprot_id", "pdb_id", "chain", "resolution", "coverage"])
        logger.info(f"SIFTS mapping: {len(uniprot_ids)} UniProt IDs -> {len(df)} PDB candidates")
        return df

    def select_best(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """
        Select the best PDB entry per UniProt ID.

        Picks the entry with highest coverage; ties are broken by
        lowest resolution.

        Args:
            candidates: DataFrame from :meth:`map_uniprot_to_pdb`.

        Returns:
            DataFrame with one row per UniProt ID.
        """
        if candidates.empty:
            return candidates

        # Sort: highest coverage first, then lowest resolution
        sorted_df = candidates.sort_values(
            by=["coverage", "resolution"],
            ascending=[False, True],
        )
        best = sorted_df.drop_duplicates(subset=["uniprot_id"], keep="first").copy()
        best = best.reset_index(drop=True)
        logger.info(f"Selected best PDB for {len(best)} UniProt IDs")
        return best
