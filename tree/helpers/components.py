#-----------------------------------------------------------------------------
# components.py:
#   Identify the unique components (monosaccharides and linkages) in strings
#   of glycans for encoding purposes.
#
#-----------------------------------------------------------------------------

from parser import parse_glycan, GlycanNode
import re

def get_components(glycan_series):
    unique_monos = set()
    unique_linkages = set()
    for glycan in glycan_series:
        root = parse_glycan(glycan)
        # traverse the tree
        stack = [root]
        while stack:
            node = stack.pop()
            if not node: 
                continue
            # If using dummy "BRANCH_ROOT", skip it as it's not a real monosaccharide
            if node.name not in ("BRANCH_ROOT", ""):
                unique_monos.add(node.name)
            for child, link in node.children:
                unique_linkages.add(link)
                stack.append(child)

    return unique_monos, unique_linkages

# Example usage:
if __name__ == "__main__":
    glycan_strings = ["Fuc(α1-2)Gal(β1-4)Glc6OS(β-Sp0", "Neu5Ac(α2-6)Gal(β1-4)GlcNAc(β1-3)Gal(β1-4)[Fuc(α1-3)]GlcNAc(β1-3)Gal(β1-4)[Fuc(α1-3)]GlcNAc(β-Sp0"]
    monos, linkages = get_components(glycan_strings)
    print("Monosaccharides:", monos)
    print("Linkage types:", linkages)

# Example output:
# Monosaccharides: {'Neu5Ac', 'Fuc', 'Gal', 'Glc6OS', 'GlcNAc'}
# Linkage types: {'α2-6', 'β1-4', 'β1-3', 'α1-3', 'α1-2'}