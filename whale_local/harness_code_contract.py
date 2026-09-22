"""Restricted executable ReTool editor output contract, without GPU imports."""
import ast

def validate_source(content):
    source = content.strip()
    if source.startswith('```python') and source.endswith('```'):
        source = source[len('```python'):-3].strip()
    tree = ast.parse(source)
    allowed_imports = {'environments.retool.base_harness', 'verifiers', 'verifiers.types',
                       'verifiers.utils.message_utils'}
    methods = {'no_tools_called', 'get_prompt_messages'}
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    assert len(classes) == 1 and classes[0].name == 'CandidateEnv'
    assert len(classes[0].bases) == 1 and isinstance(classes[0].bases[0], ast.Name) and classes[0].bases[0].id == 'ReToolEnv'
    allowed_names = {'getattr', 'hasattr', 'any', 'all', 'len', 'str', 'bool', 'reversed', 'super',
                     'UserMessage', 'concat_messages'}
    for node in tree.body:
        assert isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)), 'Only imports and CandidateEnv are allowed'
    defined_methods = set()
    for node in classes[0].body:
        assert isinstance(node, (ast.Assign, ast.AsyncFunctionDef)), 'Unexpected class statement'
        if isinstance(node, ast.Assign):
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            key = node.targets[0].id
            assert key in {'MAX_TURNS', 'SYSTEM_PROMPT', 'USER_PROMPT_TEMPLATE'}
            value = ast.literal_eval(node.value)
            if key == 'MAX_TURNS': assert isinstance(value, int) and 1 <= value <= 3
            else:
                assert isinstance(value, str) and len(value) < 2000
                if key == 'USER_PROMPT_TEMPLATE': assert value.count('{question}') == 1
        else:
            assert node.name in methods
            assert node.name not in defined_methods, 'Duplicate hook definitions'
            defined_methods.add(node.name)
            if node.name == 'no_tools_called':
                assert len(node.decorator_list) == 1 and isinstance(node.decorator_list[0], ast.Attribute)
                decorator = node.decorator_list[0]
                assert isinstance(decorator.value, ast.Name) and decorator.value.id == 'vf' and decorator.attr == 'stop'
            else:
                assert not node.decorator_list
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not any(ord(c) < 32 and c not in '\n\t\r' for c in node.value), 'Control character in code string'
            assert node.value not in {'answer', 'reward', 'ground_truth', 'target', 'is_completed', 'assistant_token_budget'}
        if isinstance(node, ast.Import): assert all(a.name in allowed_imports for a in node.names)
        if isinstance(node, ast.ImportFrom): assert node.module in allowed_imports and all(a.name != '*' for a in node.names)
        assert not isinstance(node, (ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Raise, ast.Delete, ast.Global, ast.Nonlocal))
        if isinstance(node, ast.Attribute):
            assert not node.attr.startswith('__')
            assert node.attr not in {'answer', 'reward', 'rubric', 'eval_dataset', 'dataset', 'tools', 'client'}
            if isinstance(node.ctx, ast.Store): raise ValueError('Attribute writes are not allowed')
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            assert node.slice.value not in {'answer', 'reward', 'ground_truth', 'target', 'is_completed', 'assistant_token_budget'}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id in allowed_names, f'Unexpected call: {node.func.id}'
    compile(tree, '<internal-harness>', 'exec')
    return source + '\n'
