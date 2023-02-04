class BehaviourTree:
    
    def __init__(self):
        self._root = None
        self.setup_tree()
        self._tree_execution()

    def setup_tree(self):
        pass

    def _tree_execution(self):
        while True:
            self._root.evaluate()
