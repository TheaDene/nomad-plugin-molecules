import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList
from ase.data import covalent_radii, atomic_numbers

from nomad.normalizing import Normalizer
from nomad.datamodel import EntryArchive
from nomad.datamodel.results import Material, System
from molid import query_pubchem_database


class MoleculeNormalizer(Normalizer):
    """Normalizer for molecular data extraction."""
    def normalize(self, archive: EntryArchive, logger=None) -> list:
        self.logger.info("Starting molecular normalization process.")

        if not archive.run:
            return

        try:
            # Get the atoms section from the archive
            atoms_data = archive.run[0].system[0].atoms
            atoms = self.create_atoms_object(atoms_data)

            # Split the system into connected clusters (molecules)
            clusters_indices = self.split_molecules(atoms)
            all_topology_entries = []
            # TODO: add check for system size (maybe 50-100)
            #       get_dimentionaly, only
            # If only one molecule is detected, process normally.
            if len(clusters_indices) == 1:
                print('only one molecule')
                inchikey, molecule_data = self.query_molecule_database(atoms)
                topology_entries = self.generate_topology(archive, inchikey, molecule_data)
                all_topology_entries.extend(topology_entries)
            else:
                print('multiple molecules')
                # Process each molecule cluster individually.
                for idx, indices in enumerate(clusters_indices):
                    # Create a new Atoms object for the current cluster.
                    molecule_atoms = atoms[indices]
                    inchikey, molecule_data = self.query_molecule_database(molecule_atoms)
                    topology_entries = self.generate_topology(archive, inchikey, molecule_data, molecule_id=idx+1)
                    all_topology_entries.extend(topology_entries)
            return all_topology_entries
        except Exception as e:
            self.logger.error(f"Error in normalization: {e}", exc_info=True)
            return

    def create_atoms_object(self, atoms_data) -> Atoms:
        """Creates an ASE Atoms object from NOMAD atomic data."""
        # Get the positions as a numpy array (in meters)
        atomic_positions = np.array(atoms_data.positions)
        # Convert from meters to angstrom by multiplying by 1e10
        atomic_positions_angstrom = atomic_positions * 1e10

        lattice_vectors = getattr(atoms_data, "lattice_vectors", None)
        atoms = Atoms(
            symbols=atoms_data.labels,
            positions=atomic_positions_angstrom.astype(float),
            cell=np.array(lattice_vectors, dtype=float) if lattice_vectors else None,
            pbc=atoms_data.periodic
            # pbc=[False, False, False]
        )
        return atoms

    def split_molecules(self, atoms: Atoms, scale: float = 1.2):
        """
        Splits the given ASE Atoms object into connected clusters representing individual molecules.
        Connectivity is determined by comparing interatomic distances with a cutoff
        based on the sum of covalent radii multiplied by a scaling factor.
        Returns a list of lists, each containing the indices of atoms in one molecule.
        """
        symbols = atoms.get_chemical_symbols()
        # Determine a cutoff for each atom based on its covalent radius.
        cutoffs = [covalent_radii[atomic_numbers[sym]] * scale for sym in symbols]
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)
        n_atoms = len(atoms)
        visited = [False] * n_atoms
        clusters = []
        for i in range(n_atoms):
            if not visited[i]:
                cluster_indices = []
                stack = [i]
                while stack:
                    current = stack.pop()
                    if visited[current]:
                        continue
                    visited[current] = True
                    cluster_indices.append(current)
                    indices, _ = nl.get_neighbors(current)
                    for neighbor in indices:
                        if not visited[neighbor]:
                            stack.append(neighbor)
                clusters.append(cluster_indices)
        return clusters

    def query_molecule_database(self, atoms: Atoms):
        """Queries the local PubChem database using molid."""
        database_file = '../molecule_identification/OpenBabel/pubchem_data_FULL.db'
        try:
            inchikey, molecule_data = query_pubchem_database(atoms, database_file)
            self.logger.info(f"InChIKey: {inchikey}")
            return inchikey, molecule_data
        except Exception as e:
            self.logger.error(f"Error querying the local PubChem DB: {e}")
            return None, None

    def generate_topology(self, archive, inchikey, molecule_data) -> list:
        """Returns molecular topology data in a format compatible with NOMAD."""
        if not molecule_data:
            self.logger.warning(f"No molecule data found for InChIKey {inchikey}.")
            return []

        # Ensure the archive has the results/material section available.
        print('archive.results:', archive.results)
        if not archive.results:
            print('EntryArchive().results:', EntryArchive().results)
            archive.results = EntryArchive().results
        if not archive.results.material:
            archive.results.material = archive.results.m_create(Material)
        # import pdb; pdb.set_trace()
        topology_container = archive.m_xpath('results.material.topology')
        if not topology_container:
            topology_container = archive.results.material.topology

        # Create a new System to represent the molecule topology.
        topology_entry = System(
            method='parser',
            label='molecule',
            building_block='molecule'
        )
        # Optionally, you can set additional properties (e.g. symmetry, material_id, etc.)
        # For example:
        # topology_entry.material_id = <your generated id>

        # Append the new system to the topology container.
        topology_container.append(topology_entry)

        return [topology_entry]


