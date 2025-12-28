import ast
import radon.complexity as radon_visitors

def get_functions_and_calls(file_path):
    with open(file_path, "r") as f:
        code = f.read()
    
    tree = ast.parse(code)
    
    functions = []
    calls = []
    imports = []

    # 1. FIND IMPORTS (Dependencies)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names: imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module: imports.append(node.module)

    # 2. ANALYZE FUNCTIONS
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_source = ast.get_source_segment(code, node)
            
            # A. Calculate Complexity Score (1-20)
            try:
                complexity = radon_visitors.cc_visit(func_source)[0].complexity
            except: complexity = 1
            
            # B. Detect API Routes (Flask/FastAPI decorators)
            api_route = None
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                    # Looks for @app.route('/login') or @router.get('/items')
                    if decorator.func.attr in ['route', 'get', 'post', 'put', 'delete']:
                        if decorator.args:
                            # Capture the path: "/login"
                            if isinstance(decorator.args[0], ast.Constant):
                                api_route = decorator.args[0].value # python 3.8+
                            elif isinstance(decorator.args[0], ast.Str):
                                api_route = decorator.args[0].s # python < 3.8

            # C. Detect Security Risks (Basic Heuristics)
            risks = []
            if "eval(" in func_source: risks.append("Dangerous Eval")
            if "password" in func_source.lower() and "=" in func_source: risks.append("Hardcoded Secret?")
            if "subprocess" in func_source: risks.append("Shell Injection Risk")

            # D. Detect Library Usage
            used_libs = []
            for lib in imports:
                base = lib.split('.')[0]
                if base in func_source: used_libs.append(lib)

            functions.append({
                "name": node.name,
                "code": func_source,
                "complexity": complexity,
                "libraries": used_libs,
                "api_route": api_route,   # <--- NEW
                "risks": risks            # <--- NEW
            })
            
            # E. Find Calls
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        calls.append((node.name, child.func.id))
                        
    return functions, calls