from nltk.sem import logic

read_expr = logic.Expression.fromstring


class Type:
    @property
    def is_complex(self):
        return not self.is_basic

    @property
    def is_basic(self):
        return not self.is_complex

    @classmethod
    def from_string(cls, txt: str) -> "Type":
        buffer: list[str] = list(reversed([i for i in list(txt) if i != " "]))
        stack = []

        while len(buffer):
            item = buffer.pop()
            if item in "<":
                stack.append(item)
            elif item in ">":
                y = stack.pop()
                assert len(stack) > 0
                if isinstance(stack[-1], Type):
                    x = stack.pop()
                    assert stack.pop() in "<"
                    stack.append(Complex(x, y))
                elif stack[-1] == "<" and item == ">":
                    assert stack.pop() in "<"
                    stack.append(y)
                else:
                    raise RuntimeError(f"想定外な気がする{txt}")
            elif item == ",":
                continue
            else:
                stack.append(Basic(item))

        if len(stack) == 1:
            return stack[0]
        try:
            x, y = stack
            return Complex(x, y)
        except ValueError:
            raise RuntimeError(f"falied to parse Type: {txt}")


class Basic(Type):
    def __init__(self, base: str) -> None:
        self.base: str = base

    def __str__(self) -> str:
        return self.base

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            other = Type.from_string(other)
        if isinstance(other, Complex):
            return False
        else:
            return self.base == other.base

    @property
    def is_basic(self):
        return True

    @property
    def nargs(self) -> int:
        return 0

    @property
    def depth(self) -> int:
        return 0


class Complex(Type):
    def __init__(self, left: str | Type, right: str | Type):
        self.left: Type = Type.from_string(left) if isinstance(left, str) else left
        self.right: Type = Type.from_string(right) if isinstance(right, str) else right

    def __str__(self) -> str:
        def _str(type):
            if isinstance(type, Complex):
                return f"{type}"
            return str(type)

        return "<" + _str(self.left) + "," + _str(self.right) + ">"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            other = Type.from_string(other)
        elif not isinstance(other, Complex):
            return False
        return self.left == other.left and self.right == other.right

    @property
    def is_complex(self):
        return True

    @property
    def nargs(self) -> int:
        return 1 + self.left.nargs

    @property
    def depth(self) -> int:
        return max(self.left.depth, self.right.depth) + 1


def left_is_head(left: Type, right: Type) -> bool:
    """
    両方がbasicの場合は想定していない
    """
    if isinstance(left, Type) and isinstance(right, Type):
        if left.is_complex and right.is_basic:
            return True
        elif left.is_basic and right.is_complex:
            return False
        else:
            if left.left == right:
                return True
            else:
                return False


def type2lamba(type: Type, term: str) -> logic.Expression:
    """
    0 = read_expr("0"): e, t; npとかs
    1 = read_expr("\\P.1(P)"): <e,e>, <t,t>; np/np, s/s
    2 = read_expr("\\P.\\Q.2(Q, P)"): <e,<e,t>>, <t,<e,e>>; (s/np)\\np, (np\\np)/s
    3 = read_expr("\\P.P(3)"): <<e,t>,t>; s/(s\\np)
    4 = read_expr("\\P.\\Q.4(P(Q))"): <<e,e>,<e,e>>; (np/np)/(np/np)
    conj1 = read_expr("\\P.\\Q.(Q & P)"): <e,<e,e>>; 0項のタイプ
    conj2 = read_expr("\\P.\\Q.\\R. (Q(R) & P(R))"): <<e,e>,<<e,e>,<e,e>>>; 1項のタイプ
    conj3 = read_expr("\\P.\\Q.\\R.\\S.((Q(R)(S)) & (P(R)(S)))"): <<e,<e,t>,<<e,<e,t>,<e,<e,t>>>; 2項のタイプ
    """
    if type.is_basic:
        return read_expr(term)
    elif str(type) == "<e,t>":  # VP, IV
        return read_expr(f"\\P.{term}(P)")
    elif str(type) == "<t,t>":  # that
        return read_expr("\\P.P")
    elif str(type) == "<<e,t>,t>":
        return read_expr(f"\\P.P({term})")
    elif str(type) in {"<e,<e,t>>", "<t,<e,t>>"}:
        return read_expr(f"\\P.\\Q.{term}(Q,P)")
    elif str(type) == "<<e,t>,<e,t>>":  # adv
        return read_expr(f"\\P.\\Q.{term}(P(Q))")
    elif str(type) in {"<e,<e,e>>", "<t,<t,t>>"}:
        return read_expr("\\P.\\Q.(Q & P)")
    elif str(type) in {"<<e,t>,<<e,t>,<e,t>>>", "<<<e,t>,t>,<<<e,t>,t>,<<e,t>,t>>>"}:
        return read_expr("\\P.\\Q.\\R. (Q(R) & P(R))")
    elif str(type) in {
        "<<e,<e,t>>,<<e,<e,t>>,<e,<e,t>>>>",
        "<<t,<e,t>>,<<t,<e,t>>,<t,<e,t>>>>",
        "<<<e,e>,<e,e>>,<<<e,e>,<e,e>>,<<e,e>,<e,e>>>>",
    }:
        return read_expr("\\P.\\Q.\\R.\\S.((Q(R)(S)) & (P(R)(S)))")
    else:
        raise RuntimeError(f"Unexpected condition; {str(type)=}")
