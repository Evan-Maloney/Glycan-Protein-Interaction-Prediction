#-----------------------------------------------------------------------------
# parser.py:
#   Parse the IUPAC condensed glycan data into the monosaccharides and 
#   linkages in order to obtain a tree structure.
#
# References:
#   https://github.com/demattox/glyBERT/blob/main/utils/preproccessing/sugarTrees.py
#   https://github.com/BojarLab/GIFFLAR/blob/main/convert.py
#   https://github.com/BojarLab/SweetNet/blob/main/glycowork.py
#   Used ChatGPT o3 mini to generate portions of this code and comments for clarity.
#
#-----------------------------------------------------------------------------

class GlycanNode:
    def __init__(self, name):
        self.name = name
        self.children = []

def parse_glycan(iupac):
    iupac = iupac.strip().strip('"').strip("'")
    # Remove spacer
    iupac = iupac[:iupac.rfind('(')]  

    # Recursive parser
    def _parse_seq(seq, i=0):
        node = None
        pending_children = []  
        while i < len(seq):
            if seq[i] == ']':  # end of a branch
                break  # return to caller
            if seq[i] == '[':  # start of a branch
                # parse branch recursively
                branch_node, i = _parse_seq(seq, i+1)
                if branch_node and branch_node.name == "BRANCH_ROOT":
                    pending_children.extend(branch_node.children)
                else:
                    pending_children.append((branch_node, None))
                continue
            # Parse a monosaccharide name
            j = i
            # Monosaccharide names: sequence of letters and numbers until a '(' or ')' or '[' or ']' or end
            while j < len(seq) and seq[j] not in '()[]':
                j += 1
            mono_name = seq[i:j].strip()
            if mono_name == '':
                pass
            # Create node for this monosaccharide
            current_node = GlycanNode(mono_name)
            # Attach any pending children to this node (they were waiting for this as parent)
            for child, linkage in pending_children:
                current_node.children.append((child, linkage))
            pending_children = []  # reset pending after attaching
            # Check if next part is a linkage (parentheses)
            i = j
            if i < len(seq) and seq[i] == '(':
                # linkage present
                k = seq.find(')', i)
                linkage = seq[i+1:k]  # e.g. "b1-4" or "a1-3"
                # Save current node aside as pending child with this linkage, parent unknown yet
                pending_child = (current_node, linkage)
                pending_children.append(pending_child)
                # Move index past the ')'
                i = k + 1
            else:
                # No linkage means this node is a parent or terminal in this context
                node = current_node  # this might be a root of the current (sub)chain
            # Continue parsing after this node, unless we hit end or branch close
            continue
        # End of sequence or branch reached
        # If we finish with pending_children not empty, 
        # it means these children have no parent in this context (will be attached outside).
        if pending_children:
            # If parsing main chain completely, this should not happen (there should be a final node to attach to).
            # In a branch context, this means the branch's last sugar is waiting for the main chain parent.
            # Return the first pending child (and any others) upward.
            # We'll return a tuple of a dummy node and pending list, 
            # but to keep it simple, we attach them to a new node signifying branch root.
            branch_root = GlycanNode("BRANCH_ROOT")  # dummy placeholder
            for child, linkage in pending_children:
                branch_root.children.append((child, linkage))
            return branch_root, i+1  # skip ']' and return
        return node, i+1  # return the constructed node (root of this segment) and position
    # parse the full string (main chain)
    root, _ = _parse_seq(iupac, 0)

    return root

def print_tree(node, parent=None, linkage=None, level=0):
    indent = "  " * level
    if parent and linkage:
        print(f"{indent}{node.name} --{linkage}--> {parent.name}")
    else:
        print(f"{indent}{node.name}")
    for child, link in node.children:
        print_tree(child, node, link, level+1)

# Example usage:
if __name__ == "__main__":
    glycan_str = "Neu5Ac(α2-6)Gal(β1-4)GlcNAc(β1-3)Gal(β1-4)[Fuc(α1-3)]GlcNAc(β1-3)Gal(β1-4)[Fuc(α1-3)]GlcNAc(β-Sp0"
    tree_root = parse_glycan(glycan_str)
    print_tree(tree_root)

# Example output:
# GlcNAc
#  Gal --β1-4--> GlcNAc
#    GlcNAc --β1-3--> Gal
#      Gal --β1-4--> GlcNAc
#        GlcNAc --β1-3--> Gal
#          Gal --β1-4--> GlcNAc
#            Neu5Ac --α2-6--> Gal
#      Fuc --α1-3--> GlcNAc
#  Fuc --α1-3--> GlcNAc
