"""
Data Acquisition Pipeline for Quantum Data Decoder

Handles fetching, filtering, and organizing:
- PDB structures (high-resolution, filtered by criteria)
- Kaggle competition datasets (Novozymes, CAFA 5)
- Experimental spectral databases (RamanBioLib, Sadtler, RRUFF)
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "pdb_resolution_cutoff": 2.5,  # Angstroms
    "pdb_chain_length_min": 50,
    "pdb_chain_length_max": 1000,
    "sequence_identity_cutoff": 0.90,  # 90% for PDB90
    "pdb_base_url": "https://files.rcsb.org/download/",
    "ccd_url": "https://files.rcsb.org/pub/pdb/data/monomers/",
}


class PDBDataAcquisition:
    """
    Fetch and filter PDB structures for high-quality NMA analysis.
    
    Selection criteria:
    - Resolution < 2.5 Angstroms
    - Chain length 50-1000 residues
    - Removed redundancy via sequence identity
    """
    
    def __init__(self, output_dir: str = "./data/pdb"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdb_list_url = "https://www.rcsb.org/pdb/json/querySummary?structureId="
        
    def query_pdb_advanced(self, 
                          max_resolution: float = 2.5,
                          min_length: int = 50,
                          max_length: int = 1000,
                          num_structures: int = 100) -> List[str]:
        """
        Query PDB for structures matching criteria.
        
        Note: Full RCSB API query requires advanced integration.
        This is a placeholder demonstrating the workflow.
        For production, use https://search.rcsb.org/rcsbsearch/v2/
        
        Args:
            max_resolution: Maximum resolution in Angstroms
            min_length: Minimum chain length
            max_length: Maximum chain length
            num_structures: Number of structures to retrieve
            
        Returns:
            List of PDB IDs
        """
        logger.info(f"Querying PDB for structures: resolution<{max_resolution}Å, "
                   f"chain length {min_length}-{max_length} residues")
        
        # Example: Query high-resolution NMR/X-ray structures
        # In production, integrate with RCSB Search API v2
        sample_pdb_ids = [
            '1UBQ', '1MBN', '1A1X', '1BAN', '1BAC',  # Well-characterized proteins
            '2DHB', '1HHO', '1LYZ', '2LYZ', '3LYZ',  # Hemoglobin, Lysozyme
            '1CTF', '1ELJ', '1GXN', '1HTM', '1KDH',  # Various enzymes
        ]
        
        return sample_pdb_ids[:num_structures]
    
    def download_structure(self, pdb_id: str, format: str = "pdb") -> Optional[str]:
        """
        Download a single PDB structure.
        
        Args:
            pdb_id: PDB identifier (e.g., '1UBQ')
            format: File format ('pdb' or 'mmcif')
            
        Returns:
            Path to downloaded file or None if failed
        """
        pdb_id = pdb_id.lower()
        file_ext = "pdb" if format == "pdb" else "cif"
        filename = self.output_dir / f"{pdb_id}.{file_ext}"
        
        # Skip if already downloaded
        if filename.exists():
            logger.debug(f"Structure {pdb_id} already exists")
            return str(filename)
        
        try:
            if format == "pdb":
                url = f"{CONFIG['pdb_base_url']}{pdb_id}.pdb"
            else:
                url = f"{CONFIG['pdb_base_url']}{pdb_id}.cif"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filename, 'w') as f:
                    f.write(response.text)
                logger.debug(f"Downloaded {pdb_id}")
                return str(filename)
            else:
                logger.warning(f"Failed to download {pdb_id}: Status {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error downloading {pdb_id}: {e}")
            return None
    
    def download_batch(self, pdb_ids: List[str], num_workers: int = 4) -> List[str]:
        """
        Download multiple PDB structures in parallel.
        
        Args:
            pdb_ids: List of PDB identifiers
            num_workers: Number of parallel download threads
            
        Returns:
            List of successfully downloaded file paths
        """
        downloaded_files = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.download_structure, pdb_id): pdb_id 
                      for pdb_id in pdb_ids}
            
            for future in tqdm(as_completed(futures), total=len(pdb_ids), 
                             desc="Downloading PDB structures"):
                result = future.result()
                if result:
                    downloaded_files.append(result)
        
        logger.info(f"Downloaded {len(downloaded_files)}/{len(pdb_ids)} structures")
        return downloaded_files


class KaggleDataAcquisition:
    """
    Fetch competition datasets from Kaggle using kaggle CLI.
    Requires ~/.kaggle/kaggle.json configuration.
    """
    
    def __init__(self, output_dir: str = "./data/kaggle"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_novozymes(self) -> Tuple[str, str, str]:
        """
        Download Novozymes Enzyme Stability Prediction competition data.
        
        Competition: https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction
        
        Returns:
            Tuple of (train_csv, test_csv, structure_pdb)
        """
        competition_name = "novozymes-enzyme-stability-prediction"
        logger.info(f"Downloading {competition_name} data...")

        try:
            # Use subprocess.run instead of os.system to avoid command injection
            subprocess.run(
                ["kaggle", "competitions", "download", "-c", competition_name,
                 "-p", str(self.output_dir)],
                check=True
            )
            
            # Extract if needed
            import zipfile
            for file in self.output_dir.glob("*.zip"):
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(self.output_dir)
                file.unlink()
            
            train_csv = self.output_dir / "train.csv"
            test_csv = self.output_dir / "test.csv"
            structure_pdb = self.output_dir / "wildtype_structure_prediction_af2.pdb"
            
            logger.info(f"Novozymes data downloaded to {self.output_dir}")
            return str(train_csv), str(test_csv), str(structure_pdb)
            
        except Exception as e:
            logger.error(f"Error downloading Novozymes data: {e}")
            return None, None, None
    
    def download_cafa5(self) -> Tuple[str, str]:
        """
        Download CAFA 5 Protein Function Prediction competition data.
        
        Competition: https://www.kaggle.com/competitions/cafa-5-protein-function-prediction
        
        Returns:
            Tuple of (train_csv, test_sequences_fasta)
        """
        competition_name = "cafa-5-protein-function-prediction"
        logger.info(f"Downloading {competition_name} data...")

        try:
            # Use subprocess.run instead of os.system to avoid command injection
            subprocess.run(
                ["kaggle", "competitions", "download", "-c", competition_name,
                 "-p", str(self.output_dir)],
                check=True
            )
            
            # Extract if needed
            import zipfile
            for file in self.output_dir.glob("*.zip"):
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(self.output_dir)
                file.unlink()
            
            train_csv = self.output_dir / "train_terms.csv"
            test_sequences = self.output_dir / "test_sequences.fasta"
            
            logger.info(f"CAFA 5 data downloaded to {self.output_dir}")
            return str(train_csv), str(test_sequences)
            
        except Exception as e:
            logger.error(f"Error downloading CAFA 5 data: {e}")
            return None, None
    
    def load_novozymes_data(self) -> pd.DataFrame:
        """Load and validate Novozymes training data."""
        train_csv = self.output_dir / "train.csv"
        updates_csv = self.output_dir / "train_updates.csv"
        
        # Load base training data
        df = pd.read_csv(train_csv)
        logger.info(f"Loaded base training data: {df.shape[0]} rows")
        
        # Apply updates if available (corrects data quality issues)
        if updates_csv.exists():
            updates = pd.read_csv(updates_csv)
            logger.info(f"Applying updates: {updates.shape[0]} corrections")
            # Update rows as specified in train_updates.csv
            for idx, row in updates.iterrows():
                df.loc[df['seq_id'] == row['seq_id']] = row
        
        return df
    
    def load_cafa5_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate CAFA 5 training and term data."""
        train_terms = self.output_dir / "train_terms.csv"
        train_sequences = self.output_dir / "train_sequences.fasta"
        
        terms_df = pd.read_csv(train_terms)
        logger.info(f"Loaded CAFA 5 terms: {terms_df.shape[0]} entries")
        
        # Parse sequences
        sequences = {}
        if Path(train_sequences).exists():
            with open(train_sequences, 'r') as f:
                current_id = None
                current_seq = []
                for line in f:
                    if line.startswith('>'):
                        if current_id:
                            sequences[current_id] = ''.join(current_seq)
                        current_id = line[1:].split()[0]
                        current_seq = []
                    else:
                        current_seq.append(line.strip())
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
        
        logger.info(f"Loaded {len(sequences)} protein sequences")
        return terms_df, sequences


class SpectralDatabaseAcquisition:
    """
    Ingest experimental vibrational spectral databases for validation
    and transfer learning.
    """
    
    def __init__(self, output_dir: str = "./data/spectral"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_ramanbiolib(self) -> str:
        """
        Download RamanBioLib spectral database from GitHub.
        
        Repository: https://github.com/raman-biophysics/raman-biolib
        Contains Raman spectra for amino acids, small molecules.
        """
        logger.info("Downloading RamanBioLib...")
        
        try:
            # Clone or download repository
            repo_url = "https://github.com/raman-biophysics/raman-biolib.git"
            repo_path = self.output_dir / "ramanbiolib"

            if not repo_path.exists():
                # Use subprocess.run instead of os.system to avoid command injection
                subprocess.run(
                    ["git", "clone", repo_url, str(repo_path)],
                    check=True
                )
            
            logger.info(f"RamanBioLib data stored at {repo_path}")
            return str(repo_path)
        except Exception as e:
            logger.error(f"Error downloading RamanBioLib: {e}")
            return None
    
    def generate_synthetic_spectral_db(self, num_spectra: int = 1000) -> str:
        """
        Generate synthetic spectral database for proof-of-concept.
        In production, this would be replaced with experimental data.
        
        Args:
            num_spectra: Number of synthetic spectra to generate
            
        Returns:
            Path to saved spectral database
        """
        logger.info(f"Generating {num_spectra} synthetic spectra for validation...")
        
        # Generate Lorentzian-broadened synthetic spectra
        frequencies = np.linspace(0, 500, 1000)  # 0-500 cm^-1 typical for proteins
        spectra = []
        metadata = []
        
        for i in tqdm(range(num_spectra), desc="Generating synthetic spectra"):
            # Random Lorentzian peaks simulating protein vibrational modes
            num_peaks = np.random.randint(3, 10)
            spectrum = np.zeros_like(frequencies)
            
            for _ in range(num_peaks):
                center = np.random.uniform(10, 490)
                amplitude = np.random.uniform(0.5, 2.0)
                width = np.random.uniform(2, 10)
                
                # Lorentzian profile
                lorentzian = amplitude * (width**2) / ((frequencies - center)**2 + width**2)
                spectrum += lorentzian
            
            spectra.append(spectrum)
            metadata.append({
                'spectrum_id': f'syn_{i:06d}',
                'type': 'synthetic',
                'peaks': num_peaks
            })
        
        # Save database
        spectra_array = np.array(spectra)
        db_path = self.output_dir / "synthetic_spectra_db.npz"
        np.savez(db_path, spectra=spectra_array, frequencies=frequencies, 
                 metadata=metadata)
        
        logger.info(f"Synthetic spectral database saved to {db_path}")
        return str(db_path)


def main():
    """Demonstration of data acquisition pipeline."""
    logger.info("=" * 60)
    logger.info("Quantum Data Decoder: Data Acquisition Pipeline")
    logger.info("=" * 60)
    
    # 1. PDB Data Acquisition
    logger.info("\n[Phase 1] PDB Structure Acquisition")
    pdb_acq = PDBDataAcquisition(output_dir="./data/pdb")
    pdb_ids = pdb_acq.query_pdb_advanced(num_structures=10)
    logger.info(f"Found {len(pdb_ids)} candidate structures")
    downloaded = pdb_acq.download_batch(pdb_ids, num_workers=4)
    logger.info(f"Successfully downloaded {len(downloaded)} structures\n")
    
    # 2. Kaggle Data Acquisition
    logger.info("[Phase 2] Kaggle Competition Data Acquisition")
    kaggle_acq = KaggleDataAcquisition(output_dir="./data/kaggle")
    
    # Optionally download competitions (requires kaggle CLI)
    # train_csv, test_csv, structure_pdb = kaggle_acq.download_novozymes()
    # terms_csv, sequences_fasta = kaggle_acq.download_cafa5()
    
    logger.info("(Kaggle download requires kaggle CLI credentials)\n")
    
    # 3. Spectral Database Acquisition
    logger.info("[Phase 3] Spectral Database Acquisition")
    spectral_acq = SpectralDatabaseAcquisition(output_dir="./data/spectral")
    
    # Generate synthetic spectral database
    spectral_acq.generate_synthetic_spectral_db(num_spectra=100)
    
    logger.info("\n" + "=" * 60)
    logger.info("Data acquisition pipeline complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
