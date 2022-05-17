"""
File: linkedbst.py
Author: Ken Lambert
"""
import copy
import random
import time

from binary_search_tree.abstractcollection import AbstractCollection
from binary_search_tree.bstnode import BSTNode
from binary_search_tree.linkedstack import LinkedStack
from math import log
import sys
sys.setrecursionlimit(300000)


class LinkedBST(AbstractCollection):
    """An link-based binary search tree implementation."""

    def __init__(self, sourceCollection=None):
        """Sets the initial state of self, which includes the
        contents of sourceCollection, if it's present."""
        self._root = None
        AbstractCollection.__init__(self, sourceCollection)

    # Accessor methods
    def __str__(self):
        """Returns a string representation with the tree rotated
        90 degrees counterclockwise."""

        def recurse(node, level):
            s = ""
            if node != None:
                s += recurse(node.right, level + 1)
                s += "| " * level
                s += str(node.data) + "\n"
                s += recurse(node.left, level + 1)
            return s

        return recurse(self._root, 0)

    def __iter__(self):
        """Supports a preorder traversal on a view of self."""
        if not self.isEmpty():
            stack = LinkedStack()
            stack.push(self._root)
            while not stack.isEmpty():
                node = stack.pop()
                yield node.data
                if node.right != None:
                    stack.push(node.right)
                if node.left != None:
                    stack.push(node.left)

    def preorder(self):
        """Supports a preorder traversal on a view of self."""
        return None

    def inorder(self):
        """Supports an inorder traversal on a view of self."""
        lyst = list()

        def recurse(node):
            if node != None:
                recurse(node.left)
                lyst.append(node.data)
                recurse(node.right)

        recurse(self._root)
        return iter(lyst)

    def postorder(self):
        """Supports a postorder traversal on a view of self."""
        return None

    def levelorder(self):
        """Supports a levelorder traversal on a view of self."""
        return None

    def __contains__(self, item):
        """Returns True if target is found or False otherwise."""
        return self.find(item) != None

    def find(self, item):
        """If item matches an item in self, returns the
        matched item, or None otherwise."""

        def recurse(node):
            if node is None:
                return None
            elif item == node.data:
                return node.data
            elif item < node.data:
                return recurse(node.left)
            else:
                return recurse(node.right)

        return recurse(self._root)

    # Mutator methods
    def clear(self):
        """Makes self become empty."""
        self._root = None
        self._size = 0

    def add(self, item):
        """Adds item to the tree."""

        # Helper function to search for item's position
        def recurse(node):
            # New item is less, go left until spot is found
            if item < node.data:
                if node.left == None:
                    node.left = BSTNode(item)
                else:
                    recurse(node.left)
            # New item is greater or equal,
            # go right until spot is found
            elif node.right == None:
                node.right = BSTNode(item)
            else:
                recurse(node.right)
                # End of recurse

        # Tree is empty, so new item goes at the root
        if self.isEmpty():
            self._root = BSTNode(item)
        # Otherwise, search for the item's spot
        else:
            recurse(self._root)
        self._size += 1

    def remove(self, item):
        """Precondition: item is in self.
        Raises: KeyError if item is not in self.
        postcondition: item is removed from self."""
        if not item in self:
            raise KeyError("Item not in tree.""")

        # Helper function to adjust placement of an item
        def liftMaxInLeftSubtreeToTop(top):
            # Replace top's datum with the maximum datum in the left subtree
            # Pre:  top has a left child
            # Post: the maximum node in top's left subtree
            #       has been removed
            # Post: top.data = maximum value in top's left subtree
            parent = top
            currentNode = top.left
            while not currentNode.right == None:
                parent = currentNode
                currentNode = currentNode.right
            top.data = currentNode.data
            if parent == top:
                top.left = currentNode.left
            else:
                parent.right = currentNode.left

        # Begin main part of the method
        if self.isEmpty(): return None

        # Attempt to locate the node containing the item
        itemRemoved = None
        preRoot = BSTNode(None)
        preRoot.left = self._root
        parent = preRoot
        direction = 'L'
        currentNode = self._root
        while not currentNode == None:
            if currentNode.data == item:
                itemRemoved = currentNode.data
                break
            parent = currentNode
            if currentNode.data > item:
                direction = 'L'
                currentNode = currentNode.left
            else:
                direction = 'R'
                currentNode = currentNode.right

        # Return None if the item is absent
        if itemRemoved == None: return None

        # The item is present, so remove its node

        # Case 1: The node has a left and a right child
        #         Replace the node's value with the maximum value in the
        #         left subtree
        #         Delete the maximium node in the left subtree
        if not currentNode.left == None \
                and not currentNode.right == None:
            liftMaxInLeftSubtreeToTop(currentNode)
        else:

            # Case 2: The node has no left child
            if currentNode.left == None:
                newChild = currentNode.right

                # Case 3: The node has no right child
            else:
                newChild = currentNode.left

                # Case 2 & 3: Tie the parent to the new child
            if direction == 'L':
                parent.left = newChild
            else:
                parent.right = newChild

        # All cases: Reset the root (if it hasn't changed no harm done)
        #            Decrement the collection's size counter
        #            Return the item
        self._size -= 1
        if self.isEmpty():
            self._root = None
        else:
            self._root = preRoot.left
        return itemRemoved

    def replace(self, item, newItem):
        """
        If item is in self, replaces it with newItem and
        returns the old item, or returns None otherwise."""
        probe = self._root
        while probe != None:
            if probe.data == item:
                oldData = probe.data
                probe.data = newItem
                return oldData
            elif probe.data > item:
                probe = probe.left
            else:
                probe = probe.right
        return None

    def height(self):
        """
        Return the height of tree
        :return: int
        """
        def height1(top):
            """
            Helper function
            :param top:
            :return:
            """
            if top.left == top.right == None:
                return 0
            left_sum = height1(top.left) if top.left is not None else - 1
            right_sum = height1(top.right) if top.right is not None else - 1
            return max(left_sum, right_sum) + 1
        return height1(self._root)

    def is_balanced(self):
        """
        Return True if tree is balanced
        :return:
        """
        return self.height() < 2 * log(len(self) + 1) - 1

    def range_find(self, low, high):
        """
        Returns a list of the items in the tree, where low <= item <= high."""
        elements = list(self.inorder())
        return elements[elements.index(low):elements.index(high) + 1]

    def build_rightsided_tree(self, elements):
        new_tree = LinkedBST()
        new_tree._root = BSTNode(elements.pop(0))
        curr = new_tree._root
        while len(elements) > 0:
            _next = BSTNode(elements.pop(0))
            curr.right = _next
            curr = _next
        return new_tree

    def rebalance(self):
        '''
        Rebalances the tree.
        :return:
        '''
        elements = list(self.inorder())

        def rebalance1(elems):
            if len(elems) == 0:
                return None
            i = len(elems) // 2
            elem = elems[i]
            node = BSTNode(elem)
            node.left = rebalance1(elems[:i])
            node.right = rebalance1(elems[i+1:])
            return node

        self._root = rebalance1(elements)

    def successor(self, item):
        """
        Returns the smallest item that is larger than
        item, or None if there is no such item.
        :param item:
        :type item:
        :return:
        :rtype:
        """
        elements = list(self.inorder())
        if item in elements:
            i = elements.index(item)
            if i < len(elements) - 1:
                return elements[i+1]
        else:
            if item < elements[0]:
                return elements

    def predecessor(self, item):
        """
        Returns the largest item that is smaller than
        item, or None if there is no such item.
        :param item:
        :type item:
        :return:
        :rtype:
        """
        elements = list(self.inorder())
        if item in elements:
            i = elements.index(item)
            if i > 0:
                return elements[i - 1]
        else:
            if item > elements[-1]:
                return elements[-1]

    def find_rightsided_elem(self, elem):
        curr = self._root
        while curr is not None:
            if curr == elem:
                return curr
            curr = curr.right

    def demo_bst(self, path):
        """
        Demonstration of efficiency binary search tree for the search tasks.
        :param path:
        :type path:
        :return:
        :rtype:
        """
        words = []
        with open(path, encoding='utf-8') as dictionary:
            for word in dictionary:
                words.append(word.strip().split()[0])
        words_lst = random.sample(words, 10000)
        print("1. measuring time to find 10k random words in a list...")
        start = time.time()
        for elem in words_lst:
            words.index(elem)
        end = time.time()
        print(f"*success!* it took {end-start} seconds")
        bst_dict_right = self.build_rightsided_tree(words)
        print("2. measuring time to find 10k random words in a sorted binary tree"
              "(every other element added to the right of the previous)...")
        start = time.time()
        for elem in words_lst:
            bst_dict_right.find_rightsided_elem(elem)
        end = time.time()
        print(f"*success!* it took {end - start} seconds")
        random_tree = LinkedBST()
        count = 0
        for word in random.sample(words, len(words)):
            random_tree.add(word)
            count += 1
            print(count)
        print("3. measuring time to find 10k random words in a randomly sorted binary tree...")
        start = time.time()
        for elem in words_lst:
            random_tree.find(elem)
        end = time.time()
        print(f"*success!* it took {end - start} seconds")
        random_tree.rebalance()
        print("4. measuring time to find 10k random words in a balanced binary tree...")
        start = time.time()
        for word in words_lst:
            random_tree.find(word)
        end = time.time()
        print(f"*success!* it took {end - start} seconds")


if __name__ == '__main__':
    bst = LinkedBST()
    for let in "klayobnz":
        bst.add(let)
    print(bst)
    print('_____\n')
    assert bst.height() == 4
    assert not bst.is_balanced()
    bst.rebalance()
    print(bst)
    assert (bst.is_balanced()) is True
    assert bst.successor('z') is None
    assert bst.successor('a') == 'b'
    assert bst.predecessor('a') is None
    assert bst.predecessor('z') == 'y'
    assert bst.range_find('a', 'z') == ['a', 'b', 'k', 'l', 'n', 'o', 'y', 'z']
    bst.demo_bst('words.txt')

