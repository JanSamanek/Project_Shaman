from enum import Enum


class NodeStates(Enum):
    FAILURE = 0
    SUCCESS = 1
    RUNNING = 2


class Node:

    def __init__(self, *children_nodes):

        self._data_context = {}

        self.parent = None
        self.children = []

        for child in children_nodes:
            self._attach(child)

    def _attach(self, node):
        node.parent = self
        self.children.append(node)

    def set_data(self, key, value):
        self._data_context[key] = value

    def get_data(self, key):

        value_to_get = self._data_context.get(key, None)

        if value_to_get:
            return value_to_get

        next_node_to_inspect = self.parent

        while next_node_to_inspect:
            value_to_get = next_node_to_inspect.get_data(key)
            if value_to_get:
                return value_to_get
            next_node_to_inspect = next_node_to_inspect.parent

        return None

    def clear_data(self, key):

        if key in self._data_context:
            self._data_context.pop(key)
            return True

        next_node_to_inspect = self.parent

        while next_node_to_inspect:
            cleared = next_node_to_inspect.clear_data(key)
            if cleared:
                return True
            next_node_to_inspect = next_node_to_inspect.parent

        return False

    def evaluate(self):
        return NodeStates.FAILURE


class Sequence(Node):

    def __init__(self, *children_nodes):
        super().__init__(*children_nodes)

    def evaluate(self):

        any_child_running = False

        for child in self.children:

            state = child.evaluate()

            if state == NodeStates.FAILURE:
                return NodeStates.FAILURE

            elif state == NodeStates.SUCCESS:
                continue

            elif state == NodeStates.RUNNING:
                any_child_running = True

            else:
                return NodeStates.SUCCESS

        return NodeStates.RUNNING if any_child_running else NodeStates.SUCCESS


class Fallback(Node):

    def __init__(self, *children_nodes):
        super().__init__(*children_nodes)

    def evaluate(self):

        for child in self.children:

            state = child.evaluate()

            if state == NodeStates.FAILURE:
                continue

            elif state == NodeStates.SUCCESS:
                return NodeStates.SUCCESS

            elif state == NodeStates.RUNNING:
                return NodeStates.RUNNING

            else:
                continue

        return NodeStates.FAILURE
