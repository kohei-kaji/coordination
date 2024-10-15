import re
import argparse
import copy
import os
import random
from tqdm import tqdm
from typing import Iterator, Optional

from nltk.sem.logic import Expression, ApplicationExpression, AndExpression

from type import Type, type2lamba, left_is_head

read_expr = Expression.fromstring

RANDOM_SEED: int = 42
CHANGE_TYPES: list[str] = [
    "no-reduction",
    "structure-reduction",
    "linear-reduction",
]


class Tree:
    def __init__(
        self,
        pos: str,
        children: list["Tree"] | list[str],
        parent: Optional["Tree"] = None,
        token_num: Optional[int] = None,
        sem_type: Optional[Type] = None,
        sem_expr: Optional[Expression] = None,
    ) -> None:
        assert len(children) > 0, "children must contain some Tree|str"
        self.pos = pos
        self.children = children
        self.parent = parent
        self.token_num = token_num
        self.sem_type = sem_type
        self.sem_expr = sem_expr

    def __len__(self) -> int:
        return len(self.leaves)

    @property
    def leaves(self) -> list["Tree"]:
        def _rec(node: Tree) -> None:
            if node.is_terminal:
                result.append(node)
            else:
                for child in node.children:
                    _rec(child)

        result: list["Tree"] = []
        _rec(self)
        return result

    @property
    def token(self) -> str:
        assert self.is_terminal, "Tree.token must be called on terminal objects"
        token = self.children[0]
        return token

    @property
    def tokens(self) -> list[str]:
        result: list[str] = []
        for leaf in self.leaves:
            result.append(leaf.token)
        return result

    @property
    def is_terminal(self) -> bool:
        assert self.children != [], f"An empty list is detected: pos = {self.pos}"
        return isinstance(self.children[0], str)

    def is_unary_branch(self) -> bool:
        if self.is_terminal:
            return True
        elif len(self.children) == 1:
            child = self.children[0]
            if isinstance(child, Tree):
                return child.is_unary_branch()
            else:
                return True
        else:
            return False

    def has_cc_node(self) -> bool:
        def _rec(node: Tree) -> None:
            if not node.is_terminal:
                if [child for child in node.children if child.pos == "CC"]:
                    result.append(node)
                for child in node.children:
                    _rec(child)

        result: list["Tree"] = []
        _rec(self)
        return len(result) > 0

    def has_cc_with_nnary_branches(self, n: int) -> bool:
        def _rec(node: Tree) -> None:
            if not node.is_terminal:
                if [child for child in node.children if child.pos == "CC"] and len(
                    node.children
                ) == n:
                    result.append(node)
                for child in node.children:
                    _rec(child)

        result: list["Tree"] = []
        _rec(self)
        return len(result) > 0

    def draw(self) -> str:
        def _rec(node: Tree) -> str:
            if node.is_terminal:
                return f"({node.pos} {node.token})"
            else:
                children = " ".join(_rec(child) for child in node.children)
            return f"({node.pos} {children})"

        return f"{_rec(self)}"

    def draw_terminal(self) -> str:
        result: list[str] = []
        for leaf in self.leaves:
            result.append(leaf.token)
        return " ".join(result)

    def draw_without_terminal(self) -> str:
        def _rec(node: Tree) -> str:
            if node.is_terminal:
                return ""
            else:
                children = " ".join(_rec(child) for child in node.children)
            return f"({node.pos} {children})"

        return f"{_rec(self)}"

    def update_parent(self) -> None:
        for child in self.children:
            if isinstance(child, Tree):
                child.parent = self
                child.update_parent()

    def get_gorn_addresses(self) -> list[int]:
        current_node: Tree = self
        addresses: list[int] = []
        while current_node.parent is not None:
            addresses.insert(0, current_node.parent.children.index(current_node))
            current_node: Optional[Tree] = current_node.parent
        return addresses

    def overwrite_node(self, node: "Tree", address: list[int]) -> None:
        if address:
            current_node = self
            for idx in address[:-1]:
                current_node = current_node.children[idx]

            node.parent = current_node
            current_node.children[address[-1]] = node

            # 新しいノード以下の全ノードの`parent`を更新
            self._update_children_parent(node)

    def _update_children_parent(self, node):
        for child in node.children:
            if isinstance(child, Tree):
                child.parent = node
                self._update_children_parent(child)

    def prune_cc_node(
        self, prune_right: bool, s_pos: str = "1S", already_pruned: bool = False
    ) -> bool:
        if already_pruned:
            return True

        if (
            not self.is_terminal
            and len(self.children) == 3
            and self.children[1].pos == "CC"
            and self.pos != s_pos
        ):
            if prune_right:
                self.pos = self.children[0].pos
                self.children = self.children[0].children
            else:
                self.pos = self.children[2].pos
                self.children = self.children[2].children
            return True  # 処理を行ったことを示すためにTrueを返す
        else:
            for child in self.children:
                if isinstance(child, Tree):
                    if child.prune_cc_node(prune_right, s_pos, already_pruned):
                        return True
        return False

    def get_repeated_address(self) -> list[list[int]]:
        word_set: set[str] = set()
        result: list[list[int]] = []
        for leaf in self.leaves:
            if leaf.token in word_set:
                result.append(leaf.get_gorn_addresses())
            word_set.add(leaf.token)
        return result

    def get_repeated_address_without_func(self) -> list[list[int]]:
        """
        (CC, da) (Comp, sa) (Rel, rel) (Subj, sub) (Obj, ob)
        """
        word_set: set[str] = set()
        result: list[list[int]] = []
        for leaf in self.leaves:
            if leaf.token in {"da", "sa", "rel", "sub", "ob"}:
                continue
            if leaf.token in word_set:
                result.append(leaf.get_gorn_addresses())
            word_set.add(leaf.token)
        return result

    def remove_node(self, address: list[int]) -> None:
        if address:
            current_node: Tree = self
            for gorn in address:
                if current_node.children[gorn].is_unary_branch():
                    del current_node.children[gorn]
                    break
                else:
                    current_node = current_node.children[gorn]

    def flip(self, to_flip: list[int]) -> None:
        if self.pos[0].isdigit() and int(self.pos[0]) in to_flip:
            self.children.reverse()

        for child in self.children:
            if isinstance(child, Tree):
                child.flip(to_flip)

    def reverse(self) -> None:
        self.children.reverse()
        for child in self.children:
            if isinstance(child, Tree):
                child.reverse()

    def assign_token_num(self) -> None:
        """
        各terminal nodeに、token_num（左から何番目の単語か）をふる
        一番左は1からにする（idx=0を入れると、ROOTのidxと重なるので、0は避ける）
        ついでに、各POSから数値を除去する
        """

        def _traverse(node: Tree) -> None:
            node.pos = re.sub(r"\d+", "", node.pos)
            if node.is_terminal:
                nodes.append(node)
                node.token_num = len(nodes)
            else:
                for child in node.children:
                    _traverse(child)

        nodes: list[Tree] = []
        _traverse(self)

    def cc_is_in_rightspine(self) -> bool:
        rightspines: list[Tree] = []

        def _traverse(tree: Tree) -> None:
            if not tree.is_terminal:
                _traverse(tree.children[-1])
                rightspines.append(tree)

        _traverse(self)
        for node in rightspines:
            if (
                node.children
                and len(node.children) == 3
                and node.children[1].pos == "CC"
            ):
                return True
        return False


def expand_cc(whole_tree: Tree, s_pos: str = "1S") -> Tree:
    def _expand(node: Tree) -> bool:
        if node.is_terminal:
            return False

        for child in node.children:
            if isinstance(child, Tree):
                if (
                    len(child.children) == 3
                    and child.children[1].pos == "CC"
                    and child.pos != s_pos
                ):
                    cc: Tree = copy.deepcopy(child.children[1])
                    current: Tree = child
                    while current is not None and current.pos != s_pos:
                        child: Tree = current
                        current: Optional[Tree] = current.parent
                    if current:
                        address: list[int] = current.get_gorn_addresses()
                        left: Tree = copy.deepcopy(current)
                        left.prune_cc_node(prune_right=True)
                        right: Tree = copy.deepcopy(current)
                        right.prune_cc_node(prune_right=False)
                        new_s_node = Tree(s_pos, [left, cc, right], current.parent)
                        whole_tree.overwrite_node(new_s_node, address)
                        return True
                    return False
                else:
                    if _expand(child):
                        return True
        return False

    while _expand(whole_tree):
        pass  # CCノードに対する変更が全て完了するまでループ

    return whole_tree


def annotate_semantics(tree: Tree) -> None:
    def _assign_sem(node: Tree) -> None:
        if node.is_terminal:
            node.pos = re.sub(r"\d+", "", node.pos)
            if node.pos.startswith("Noun") or node.pos.startswith("Pronoun"):
                node.sem_type = Type.from_string("e")
            elif node.pos == "Adj":
                node.sem_type = Type.from_string("<e,e>")
            elif node.pos.startswith("IVerb"):
                node.sem_type = Type.from_string("<e,t>")
            elif node.pos == "Comp":
                node.sem_type = Type.from_string("<t,t>")
            elif node.pos == "Prep":
                node.sem_type = Type.from_string("<e,<e,e>>")
            elif node.pos.startswith("TVerb"):
                node.sem_type = Type.from_string("<e,<e,t>>")
            elif node.pos.startswith("Verb_Comp"):
                node.sem_type = Type.from_string("<t,<e,t>>")
            elif node.pos == "CC":
                node.sem_type = Type.from_string("<t,<t,t>>")
            elif node.pos == "Rel":
                node.sem_type = Type.from_string("<<e,t>,<e,e>>")
            elif node.pos in {"Subj", "Obj"}:
                node.sem_type = Type.from_string("<e,e>")
                node.sem_expr = read_expr("\\P.P")
            else:
                print(node)
            if node.sem_expr is None:
                node.sem_expr = type2lamba(node.sem_type, node.token)
        else:
            for child in node.children:
                _assign_sem(child)

    def _update_semantics(node: Tree) -> None:
        """
        Recursively update the semantic representation based on the child nodes.
        """
        if node.is_terminal:
            return

        for child in node.children:
            _update_semantics(child)

        if len(node.children) == 1:
            node.sem_type = node.children[0].sem_type
            node.sem_expr = node.children[0].sem_expr

        elif len(node.children) == 2:
            left_child = node.children[0]
            right_child = node.children[1]
            if left_is_head(left_child.sem_type, right_child.sem_type):
                node.sem_type = left_child.sem_type.right
                node.sem_expr = left_child.sem_expr(right_child.sem_expr).simplify()
            else:
                node.sem_type = right_child.sem_type.right
                node.sem_expr = right_child.sem_expr(left_child.sem_expr).simplify()

        elif len(node.children) == 3:
            # head-final treeを想定し、left-branchとする
            left_child = node.children[0]
            mid_child = node.children[1]
            right_child = node.children[2]
            midP_sem_expr = mid_child.sem_expr(left_child.sem_expr).simplify()
            node.sem_expr = midP_sem_expr(right_child.sem_expr).simplify()
            node.sem_type = right_child.sem_type

    _assign_sem(tree)
    _update_semantics(tree)


class PTBReader:
    def __init__(self, line: str) -> None:
        self.line: str = line.replace("( ", "(").replace(" )", ")")
        self.index: int = 0
        self.tokens = []

    def _next(self, target: str) -> str:
        start: int = self.index
        end: int = self.line.find(target, self.index)
        result: str = self.line[start:end]
        self.index: int = end + 1
        return result

    def _is_current_idx(self, text: str) -> None:
        if self.line[self.index] != text:
            raise RuntimeError("the position of 'index' is not correct")

    def peek_current_index_str(self) -> str:
        return self.line[self.index]

    def parse(self) -> Tree:
        return self._next_node()

    @property
    def _next_node(self):
        end = self.line.find(" ", self.index)
        if self.line[end + 1] == "(":
            return self.parse_tree
        else:
            return self.parse_terminal

    def parse_terminal(self) -> Tree:
        self._is_current_idx("(")
        pos = self._next(" ")[1:]
        token = self._next(")")
        self.tokens.append(token)
        return Tree(pos, [token])

    def parse_tree(self) -> Tree:
        self._is_current_idx("(")
        pos = self._next(" ")[1:]
        self._is_current_idx("(")

        children = []
        while self.peek_current_index_str() != ")":
            children.append(self._next_node())
            if self.peek_current_index_str() == " ":
                self._next(" ")

        self._next(")")

        return Tree(pos, children)


def reader(filepath: str) -> Iterator[Tree]:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f):
            # if "... )" in line:  # 長すぎによるエラー回避用
            #     continue
            tree: Tree = PTBReader(line).parse()
            tree.update_parent()
            yield tree


def change_tree_structure(tree: Tree, change_type: str) -> Tree:
    assert change_type in CHANGE_TYPES
    tree = copy.deepcopy(tree)
    if change_type == "no-reduction":
        expanded: Tree = expand_cc(tree)
        return expanded
    elif change_type == "structure-reduction":
        return tree
    elif change_type == "linear-reduction":
        tree = expand_cc(tree)
        repeated_addresses = tree.get_repeated_address()
        if repeated_addresses:
            for address in reversed(repeated_addresses):
                tree.remove_node(address)
        return tree
    else:
        raise ValueError(f"Unhandled change_type: {change_type}")


def make_data_of_artificial_langs(
    permutated_sentences_dir: str,
    output_dir: str,
    num_of_sents: int,
    random_seed: int = RANDOM_SEED,
) -> None:

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    random.seed(random_seed)
    indices: list[int] = list(range(num_of_sents))
    random.shuffle(indices)

    for path in tqdm(os.listdir(permutated_sentences_dir)):
        if not path.endswith(".txt"):
            continue
        permutated_sentences_file: str = os.path.join(permutated_sentences_dir, path)
        trees: list[Tree] = list(reader(permutated_sentences_file))
        trn_end: int = int(num_of_sents * 0.8)
        dev_end: int = int(num_of_sents * 0.9)

        grammar_name: str = path.split(".")[0]
        subdir_path: str = os.path.join(output_dir, grammar_name)
        os.makedirs(subdir_path)
        for change_type in CHANGE_TYPES:
            change_type_dir: str = os.path.join(subdir_path, change_type)
            os.makedirs(change_type_dir)
            trn_trees = []
            trn_tokens = []
            dev_trees = []
            dev_tokens = []
            tst_trees = []
            tst_tokens = []

            random_sorted_trees = [trees[i] for i in indices]
            for i, tree in enumerate(random_sorted_trees):
                transformed_tree: Tree = change_tree_structure(
                    tree, change_type=change_type
                )
                if i < trn_end:
                    trn_trees.append(transformed_tree.draw())
                    trn_tokens.append(transformed_tree.draw_terminal())
                elif i < dev_end:
                    dev_trees.append(transformed_tree.draw())
                    dev_tokens.append(transformed_tree.draw_terminal())
                else:
                    tst_trees.append(transformed_tree.draw())
                    tst_tokens.append(transformed_tree.draw_terminal())

            with open(os.path.join(change_type_dir, "trn.tree"), "w") as trn_tree:
                trn_tree.write("\n".join(trn_trees))
            with open(os.path.join(change_type_dir, "trn.token"), "w") as trn_token:
                trn_token.write("\n".join(trn_tokens))
            with open(os.path.join(change_type_dir, "dev.tree"), "w") as dev_tree:
                dev_tree.write("\n".join(dev_trees))
            with open(os.path.join(change_type_dir, "dev.token"), "w") as dev_token:
                dev_token.write("\n".join(dev_tokens))
            with open(os.path.join(change_type_dir, "tst.tree"), "w") as tst_tree:
                tst_tree.write("\n".join(tst_trees))
            with open(os.path.join(change_type_dir, "tst.token"), "w") as tst_token:
                tst_token.write("\n".join(tst_tokens))


def permutate_sentences(sample_sentence_file: str, output_dir: str) -> int:
    def _permutate_sentence_file(
        i: int, sentences: list[str], output_file: str
    ) -> None:
        with open(output_file, "w") as f_out:
            for sentence in sentences:
                tree: Tree = PTBReader(sentence).parse()
                to_flip: list[int] = [j + 1 for j in range(6) if (i >> j) & 1 == 1]
                tree.flip(to_flip)
                print(tree.draw(), file=f_out)

    with open(sample_sentence_file, "r") as file:
        sentences: list[str] = file.readlines()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i in range(64):
        grammar_name = format(i, "06b")[::-1]
        output_file = os.path.join(output_dir, grammar_name + ".txt")
        _permutate_sentence_file(i, sentences, output_file)

    return len(sentences)


def make_logical_expression(treebank_path: str, output_path: str) -> None:
    with open(output_path, "w") as f:
        for tree in tqdm(reader(treebank_path)):
            annotate_semantics(tree)
            print(tree.sem_expr, file=f)


def change_representation(expr):
    if isinstance(expr, ApplicationExpression):
        if len(expr.args) == 2:
            return f"{change_representation(expr.function.function)} {change_representation(expr.args[0])} {change_representation(expr.args[1])} {change_representation(expr.function.function)}"
        elif len(expr.args) == 1:
            return f"{change_representation(expr.function)} {change_representation(expr.argument)} {change_representation(expr.function)}"
    elif isinstance(expr, AndExpression):
        return f"( {change_representation(expr.first)} AND {change_representation(expr.second)} )"
    else:
        return str(expr)


if __name__ == "__main__":
    # line = "(ROOT (1S (NP_Subj_S (NP_S (Noun_S madetor)) (Subj sub)) (VP_S (VP_Pres_S (IVerb_Pres_S tirastes)))))"
    # tree = PTBReader(line).parse()
    # print(tree.draw_without_terminal())
    # noun = read_expr("noun")
    # tverb = read_expr("\P Q.verb(Q,P)")
    # rel = read_expr("(\P Q.(Q & P(Q)))")
    # head_noun = read_expr("headN")
    # vp = tverb(noun).simplify()
    # print(vp)
    # r = rel(vp).simplify()
    # print(r)
    # n = r(head_noun).simplify()
    # print(n)

    # line = "(ROOT (1S (NP_Subj_P (5NP_P (Adj (Adj skame) (CC da) (Adj skeer)) (Noun_P fossicians)) (Subj sub)) (VP_P (VP_Comp_P (2VP_Comp_Past_P (3S_Comp (1S (NP_Subj_P (NP_P (Pronoun_P si)) (Subj sub)) (VP_P (2VP_Past_P (NP_Obj (6NP_S (VP_S (2VP_Past_S (NP_Obj (6NP_S (VP_S (VP_Comp_S (2VP_Comp_Pres_S (3S_Comp (1S (NP_Subj_P (NP_P (Noun_P shodderists)) (Subj sub)) (VP_P (2VP_Pres_P (NP_Obj (6NP_P (VP_P (VP_Past_P (IVerb_Past_P teeveda))) (Rel rel) (Noun_P redullors)) (Obj ob)) (TVerb_Pres_P heferate)))) (Comp sa)) (Verb_Comp_Pres_S saunches)))) (Rel rel) (Noun_S stremorder)) (Obj ob)) (TVerb_Past_S maffliced))) (Rel rel) (Noun_S kuestor)) (Obj ob)) (TVerb_Past_P bolicizeda)))) (Comp sa)) (Verb_Comp_Past_P craileda))))))"
    # tree = PTBReader(line).parse()
    # annotate_semantics(tree)
    # for leaf in tree.leaves:
    #     print(leaf.token)
    #     print(leaf.sem_type)
    #     print(leaf.sem_expr)
    # print(tree.sem_expr)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample_sentence_file",
        type=str,
        required=True,
        help="Path to sample sentence file",
    )
    parser.add_argument(
        "--permutated_output_dir",
        type=str,
        required=True,
        help="Location of output dir of permutated samples",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        required=True,
        help="Location of output dir of splitted_samples",
    )

    args = parser.parse_args()

    num_of_sents: int = permutate_sentences(
        args.sample_sentence_file, args.permutated_output_dir
    )
    make_data_of_artificial_langs(
        args.permutated_output_dir, args.split_dir, num_of_sents
    )
