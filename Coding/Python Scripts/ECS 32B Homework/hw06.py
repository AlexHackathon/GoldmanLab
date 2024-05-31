
class BinaryTreeNode:
    def __init__(self, data):
        self.val = data
        self.left = None
        self.right = None
    def setLeft(self,node):
        self.left = node
    def setRight(self,node):
        self.right = node


"""
1. Iterative preorder traversal of a binary tree
"""
def preorder(root):
    if root == None:
        return []
    roots = [root]
    preorder = []
    while len(roots) > 0:
        if roots[-1].left==None and roots[-1].right == None:
            x = roots.pop()
            preorder.append(x.val)
        elif roots[-1].left == None and roots[-1].right!=None:
            #Adds the existing right node to the stack
            x = roots.pop()
            print("Next" + str(x.val))
            roots.append(x.right)
            preorder.append(x.val)
            print("Order: " + str(preorder))
        elif roots[-1].right == None and roots[-1].left!=None:
            #Adds the existing left node to the stack
            x = roots.pop()
            print("Next" + str(x.val))
            roots.append(x.left)
            preorder.append(x.val)
            print("Order: " + str(preorder))
        else:
            #Remove the last node on the stack and adds the children (right first)
            x = roots.pop()
            print("Next" + str(x.val))
            roots.append(x.right)
            roots.append(x.left)
            preorder.append(x.val)
            print("Order: " + str(preorder))
        print("Stack: " + str([v.val for v in roots]))
        print(len(roots) > 0)
    print("Escape")
    return preorder
"""
2. Reconstruct Binary Tree
"""
# inorder traversal can be divided as [left-subtree-nodes, root, right-subtree-nodes]
# preorder traversal can be divided as [root, left-subtree-nodes, right-subtree-nodes]
def reconstructBT(preorder, inorder):
    #The first preorder is the root
    #Partition the preorder at the value before the root in the inorder traversal
    #Recursively call the function with the new preorder and inorder traversal for the subtrees
    #Base case: the traversal is a single node so return the node
    if len(preorder) == 0:
        return None
    if len(preorder) == 1:
        return BinaryTreeNode(preorder[0])
    else:
        inIdx = inorder.index(preorder[0])
        root = BinaryTreeNode(preorder[0])
        try:
            root.setLeft(reconstructBT(preorder[1:inIdx+1],inorder[:inIdx]))
        except:
            root.setLeft(None)
        try:
            root.setRight(reconstructBT(preorder[inIdx+1:],inorder[inIdx+1:]))
        except:
            root.setRight(None)
        return root

"""
3. Convert Binary Search Tree
"""
#Convert the list to a reverse inorder list
#Compute the sums in another list
#Create a list of the nodes
#Set the node values as the values at corresponding indices of the sum list
def convertToList(root):
    #Use reverse inorder traversal to create a list from greatest to least
    if root == None:
        return []
    else:
        return convertToList(root.right) + [root.val] + convertToList(root.left)
def summationList(list):
    #Use the list of greatest to least to calculate a list of the sum of all greater elements
    mySum = 0
    for i in range(len(list)):
        mySum += list[i]
        list[i] = mySum
    return list
def iterativeReverseInorder(node):
    #Uses iterative reverse inorder traversal to visit the nodes and create a list of nodes in the order of the traversal
    nodeStack = []
    traversalList = []
    nodeStack.append(node)
    curr = node
    #Go all the way down the right branch adding to the stack
    while len(nodeStack) != 0:
        while curr != None:
            if curr.right != None:
                nodeStack.append(curr.right)
            curr = curr.right
        #When the end is hit, add to the traversal list until a potential left branch is discovered
        x = nodeStack.pop()
        traversalList.append(x)
        #Go down the right of the left branch again adding to the stack
        if x.left != None:
            nodeStack.append(x.left)
            curr = x.left
    #Return the list and a pointer to the head node
    return (traversalList, node)
def finalConvert(sumList, nodeList, head):
    #Go over every node in the order of the inorder traversal to
    for i in range(len(sumList)):
        nodeList[i].val = sumList[i]
    return head
def convertBSTtoGST(root):
    if root == None:
        return None
    x = iterativeReverseInorder(root)
    print([e.val for e in x[0]])
    y = convertToList(root)
    print(y)
    z = summationList(y)
    print(z)
    return finalConvert(z, x[0], x[1])
