from dataclasses import fields, is_dataclass

from sci_utils import validation as validation


def traverse_dotted_path(root, dotted_path: str):

    validation.validate_str(dotted_path)
    if dotted_path == '':
        return root

    dotted_path_parts = dotted_path.split('.')

    branch = root
    for i_part, part in enumerate(dotted_path_parts):
        # Validate during iteration so exact failure point can be output.
        if part == '':
            raise ValueError(
                f"'' detected for {part} at position {i_part} in dotted path: "
                f"'{dotted_path}'. Check for double dot typos, e.g. 'a..b.c'."
            )
        if part.strip() != part:
            raise ValueError(
                f"Empty space detected in for {part} at position {i_part} in "
                f"dotted_path: '{dotted_path}'."
            )
        if branch is None:
            # Raise exception if None encountered on a non-leaf node.
            raise ValueError(
                "None value encountered for branch when attempting to access with "
                f"{part} at position {i_part} in dotted_path: '{dotted_path}'."
            )

        if is_dataclass(branch):
            branch = getattr(branch, part)
        elif isinstance(branch, dict):
            branch = branch[part]
        elif isinstance(branch, (list, tuple)):
            # try:
            #     idx = int(part)
            #     validation_utils.validate_nonneg_int(idx)
            # except ValueError:
            #     raise ValueError(
            #         "Expected a str representation of a non-negative integer at "
            #         f"position {i_part} in dotted path: '{dotted_path}', but got {part}."
            #     )
            # branch = branch[part]
            if part == '*':
                # Wildcard, traverse for all items in iterable.
                remainder = '.'.join(dotted_path_parts[i_part+1:])
                if remainder == '':
                    # If no more parts, return the whole iterable.
                    return list(branch) if isinstance(branch, tuple) else branch
                else:
                    # Otherwise, recursively traverse the rest of the path.
                    return [traverse_dotted_path(item, remainder) for item in branch]
            else:
                # If part is not wildcard, assume it is an index.
                try:
                    idx = int(part)
                except ValueError:
                    raise ValueError(
                        "Expected a str representation of an int at position "
                        f"{i_part} in dotted path: '{dotted_path}', but got {part}."
                    )
                try:
                    branch = branch[idx]
                except IndexError:
                    raise IndexError(
                        f"Index {idx} out of bounds for branch at position {i_part} in "
                        f"dotted path: '{dotted_path}'. Branch has length {len(branch)}."
                    )
                # try:
                #     part = ast.literal_eval(part)
                # except (ValueError, SyntaxError):
                #     raise ValueError(
                #         f"Expected a str representation of a list of non-negative integers at "
                #         f"position {i_part} in dotted path: '{dotted_path}', but got {part}."   
                #     )
                # branch = [traverse_dotted_path(item, '.'.join(dotted_path_parts[i_part+1:])) for item in branch[part]]
        else:
            incorrect = 'root' if i_part == 0 else f'branch {i_part-1}'
            raise RuntimeError(
                "Unexpected condition reached during attempted traversal of "
                f"root object according to dotted path: '{dotted_path}'. "
                "Expected a dataclass, dict, or iterable for root and all branches "
                f"but got type {type(branch)} for {incorrect}. This resulted in "
                f"this exception being raised at position {i_part} in the dotted path."
            )

    return branch


def shallow_asdict(d):
    """ 
    """
    if not is_dataclass(d):
        raise TypeError("`shallow_asdict` should be called on dataclass instances.")
    return {f.name : getattr(d, f.name) for f in fields(d)}