package assist;

public class UnionFind implements UF {
    private int[] rank;   // rank[i]粗略地表示以i为根的集合所表示的树的层数（因为路径被压缩了）
    private int[] parent; // parent[i]表示第i个元素所指向的父节点
    int size = 0;

    public UnionFind(char[][] grid) {
        int rowLen = grid.length;
        int colLen = grid[0].length;
        for (int i = 0; i < rowLen; ++i) {
            for (int j = 0; j < colLen; ++j) {
                if (grid[i][j] == '1') {
                    parent[i * colLen + j] = i * colLen + j;
                    size++;
                }
                rank[i * colLen + j] = 1;
            }
        }
    }

    public UnionFind(int size) {
        parent = new int[size];
        rank = new int[size];
        for (int i = 0; i < size; i++) {
            parent[i] = i;
            rank[i] = 1;
        }
    }

    @Override
    public int getSize() {
        return size;
    }

    @Override
    //时间复杂度O(h)
    public boolean isConnected(int p, int q) {
        return find(p) == find(q);
    }

    @Override
    //时间复杂度O(h)
    public void unionElements(int p, int q) {
        int pRoot = find(p);
        int qRoot = find(q);
        if (pRoot == qRoot) return;
        if (rank[pRoot] < rank[qRoot]) {
            parent[pRoot] = qRoot;
        } else if (rank[qRoot] < rank[pRoot]) {
            parent[qRoot] = pRoot;
        } else {
            parent[qRoot] = pRoot;
            rank[pRoot] += 1;
        }
        --size;
    }

    //时间复杂度O(h)
    private int find(int p) {
        if (p < 0 || p >= parent.length) throw new IllegalArgumentException("p is out of bound.");
        // 不断去查询自己的父亲节点, 直到到达根节点
        // 根节点的特点: parent[p] == p
        if (p != parent[p]) {
            parent[p] = find(parent[p]);
        }
        return parent[p];
    }
}
