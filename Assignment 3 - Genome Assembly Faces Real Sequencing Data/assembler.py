# python3
class DeBruijnAssembler:
    def __init__(self, reads):
        self.reads = reads
        self.graph = {}  # De Bruijn graph as a dictionary

    def construct_de_bruijn_graph(self, k):
        # Build the De Bruijn graph from reads
        for read in self.reads:
            for i in range(len(read) - k + 1):
                kmer = read[i:i + k]
                prefix = kmer[:-1]
                suffix = kmer[1:]
                if prefix not in self.graph:
                    self.graph[prefix] = [suffix]
                else:
                    self.graph[prefix].append(suffix)

    def traverse_graph(self):
        # Traverse the De Bruijn graph to find contigs
        contigs = []
        for node in self.graph:
            if len(self.graph[node]) == 1:  # Handle branching
                contig = node
                while len(self.graph[contig[-1]]) == 1:
                    contig += self.graph[contig[-1]][0][-1]
                contigs.append(contig)
        return contigs

    def assemble_genome(self, k):
        self.construct_de_bruijn_graph(k)
        contigs = self.traverse_graph()
        return contigs

# Example usage:
reads = ["ATGCTAGAC", "TAGACGCTA", "CTAGACGCT"]
assembler = DeBruijnAssembler(reads)
contigs = assembler.assemble_genome(3)

# Print contigs in FASTA format
for i, contig in enumerate(contigs):
    print(f">Contig{i+1}")
    print(contig)
