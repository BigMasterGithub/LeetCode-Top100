import assist.ListNode;
import assist.TreeNode;
import assist.UnionFind;
import cn.hutool.core.lang.tree.Tree;
import cn.hutool.core.util.NumberUtil;
import utils.Util;

import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/2/2 18:31
 **/
public class Solution {


    // 1. 两数之和
    public int[] twoSum(int[] nums, int target) {
        if (nums == null || nums.length < 2) return null;
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[]{i, map.get(target - nums[i])};
            }
            map.put(nums[i], i);
        }
        return null;
    }

    // 2. 两数相加
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {

        if (l1 == null || l2 == null) return l1 == null ? l2 : l1;

        ListNode ans = new ListNode(0);

        ListNode p = l1;
        ListNode q = l2;
        ListNode cur = ans;
        int temp = 0;
        int remainder = 0;
        //当 pq都走到头时退出
        while (p != null || q != null) {
            int v1 = p == null ? 0 : p.val;
            int v2 = q == null ? 0 : q.val;
            remainder = (v1 + v2 + temp) % 10;
            temp = (v1 + v2 + temp) / 10;

            ListNode node = new ListNode(remainder);
            cur.next = node;
            cur = cur.next;
            p = (p == null) ? null : p.next;
            q = (q == null) ? null : q.next;
        }
        if (temp == 1) cur.next = new ListNode(1);
        return ans.next;
    }

    // 3 无重复字符的最长子串
    public int lengthOfLongestSubstring(String s) {

        if (s == null || s == "" || s.length() == 0) return 0;
        int l = 0;
        int r = 0;
        int curMaxLength = 0;
        // map 记录遍历时字符出现的最新位置.
        Map<Character, Integer> map = new HashMap<>();

        for (int i = 0; i < s.length(); i++) {
            r = i;
            Character c = s.charAt(i);
            if (map.containsKey(c)) {
                // 一旦窗口中有重复的内容出现,就修改左边界l,将l置为 map中该字符的位置的下一位.
                l = Math.max(l, map.get(c) + 1);

            }
            map.put(c, i);
            curMaxLength = Math.max(curMaxLength, r - l + 1);
        }
        return curMaxLength;
    }

    // 4. 寻找两个正序数组的中位数
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {

        int length1 = nums1.length;
        int length2 = nums2.length;
        if (NumberUtil.isOdd(length1 + length2)) {
            int kth = (length1 + length2) / 2 + 1;
            return kthNumber(nums1, nums2, kth);
        } else {
            int temp1 = kthNumber(nums1, nums2, (length2 + length1) / 2);
            int temp2 = kthNumber(nums1, nums2, (length2 + length1) / 2 + 1);
            return (double) (temp1 + temp2) / 2;
        }

    }

    // 返回两个正序数组中第k小的数
    private int kthNumber(int nums1[], int nums2[], int k) {
        int len1 = nums1.length;
        int len2 = nums2.length;
        int start1 = 0;
        int start2 = 0;
        while (true) {
            if (start1 == len1) return nums2[start2 + k - 1];
            if (start2 == len2) return nums1[start1 + k - 1];

            if (k == 1) return Math.min(nums1[start1], nums2[start2]);

            int half = k / 2;
            int p = Math.min(len1, start1 + half) - 1;


            int q = Math.min(len2, start2 + half) - 1;

            if (nums1[p] > nums2[q]) {
                k -= (q - start2 + 1);
                start2 = q + 1;
            } else {
                k -= (p - start1 + 1);
                start1 = p + 1;
            }


        }

    }

    // 5.最长回文子串
    public String longestPalindrome(String s) {
        int len = s.length();
        boolean dp[][] = new boolean[len][len];
        for (int i = 0; i < len; i++)
            dp[i][i] = true;
        int maxLen = 1;
        int l = 0;
        char arr[] = s.toCharArray();
        for (int step = 2; step <= len; step++) {
            for (int L = 0; L < len; L++) {
                int R = L + step - 1;
                if (R >= len) break;
                if (arr[L] != arr[R]) dp[L][R] = false;
                else {//arr[L] == arr[R]
                    if (R == L + 1 || R == L + 2) dp[L][R] = true;
                    dp[L][R] = dp[L + 1][R - 1];
                }

                if (dp[L][R] && step > maxLen) {
                    maxLen = step;
                    l = L;
                }
            }
        }
        return s.substring(l, l + maxLen);
    }


    //11. 盛最多水的容器
    public int maxArea(int[] height) {
        int L = 0;
        int R = height.length - 1;
        int ans = 0;
        int temp = 0;
        while (L < R) {
            if (height[L] < height[R]) {
                temp = (R - L) * height[L];
                L++;
            } else { //height[L] 大于或等于 height[R]
                temp = (R - L) * height[R];
                R--;
            }
            ans = Math.max(ans, temp);
        }
        return ans;
    }

    // 15. 三数之和
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();

        int length = nums.length;
        Arrays.sort(nums);
        if (nums[0] > 0) return null;
        for (int one = 0; one < length; one++) {
            if (one > 0 && nums[one] == nums[one - 1]) continue;
            int temp = -nums[one];
            int three = length - 1;
            for (int two = one + 1; two < length; two++) {
                if (two > three + 1 && nums[two] == nums[two - 1]) continue;

                while (two < three && nums[two] + nums[three] > temp) three--;
                if (two == three) break;
                if (nums[two] + nums[three] == temp) ans.add(Arrays.asList(nums[one], nums[two], nums[three]));
            }
        }


        return ans;
    }

    // 17. 电话号码的字母组合
    public List<String> letterCombinations(String digits) {
        List<String> ans = new ArrayList<>();
        Map<Character, String> map = new HashMap<>();
        if (digits.length() == 0) return ans;
        map.put('2', "abc");
        map.put('3', "def");
        map.put('4', "ghi");
        map.put('5', "jkl");
        map.put('6', "mno");
        map.put('7', "pqrs");
        map.put('8', "tuv");
        map.put('9', "wxyz");

        findAns(ans, new StringBuilder(), digits, map, 0);
        return ans;
    }

    private void findAns(List<String> ans, StringBuilder temp, String digits, Map<Character, String> map, int index) {
        if (digits.length() == index) {
            ans.add(temp.toString());
            return;
        }
        String curr = map.get(digits.charAt(index));
        for (int i = 0; i < curr.length(); i++) {
            temp.append(curr.charAt(i));
            findAns(ans, temp, digits, map, index + 1);
            temp.deleteCharAt(index);
        }
    }


    //19. 删除链表的倒数第 N 个结点  方法一 常规思路法
    public ListNode removeNthFromEnd(ListNode head, int n) {
        int length = 0;
        ListNode cur = head;

        while (cur != null) {
            length++;
            cur = cur.next;
        }
        cur = head;

        if (n == length) return cur.next;
        for (int i = 1; i < length - n; i++) {
            cur = cur.next;
        }
        cur.next = cur.next.next;
        return head;
    }


    //19. 删除链表的倒数第 N 个结点  方法二  双指针法
    public ListNode removeNthFromEnd2(ListNode head, int n) {
        ListNode fast = head;
        ListNode slow = head;

        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        //特殊情况:  删除的结点为 头结点
        if (fast == null) return head.next;

        while (fast != null && slow != null) {
            fast = fast.next;

            slow = slow.next;
        }
        slow.next = slow.next.next;
        return head;
    }

    // 20. 有效的括号
    public boolean isValid(String s) {
        char[] arr = s.toCharArray();
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == '(') stack.push(')');
            else if (arr[i] == '[') stack.push(']');
            else if (arr[i] == '{') stack.push('}');
            else {
                if (stack.isEmpty() || stack.peek() != arr[i]) {
                    return false;
                }
                stack.pop();
            }
        }
        return stack.isEmpty();
    }

    // 21. 合并两个有序链表
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if (list1 == null || list2 == null) return list1 == null ? list1 : list2;

        ListNode cur1 = list1;
        ListNode cur2 = list2;
        ListNode ansHead = new ListNode(0);
        ListNode cur = ansHead;
        while (cur1 != null && cur2 != null) {
            if (cur1.val < cur2.val) {

                cur.next = cur1;
                cur = cur.next;
                cur1 = cur1.next;
            } else {
                cur.next = cur2;
                cur = cur.next;
                cur2 = cur2.next;
            }

        }
        if (cur1 != null) cur.next = cur1;
        if (cur2 != null) cur.next = cur2;
        return ansHead.next;
    }

    // 22. 括号生成
    public List<String> generateParenthesis(int n) {
        List<String> ans = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        backTrack(ans, cur, 0, 0, n);
        return ans;

    }

    private void backTrack(List<String> ans, StringBuilder cur, int leftSum, int rightSum, int leftMax) {
        if (cur.length() == 2 * leftMax) {
            ans.add(cur.toString());
        }
        // 左边的括号 最多能有 n 个.
        if (leftSum < leftMax) {
            cur.append('(');
            backTrack(ans, cur, leftSum + 1, rightSum, leftMax);
            cur.deleteCharAt(cur.length() - 1);
        }
        // 添加 右边的括号
        if (rightSum < leftSum) {
            cur.append(')');
            backTrack(ans, cur, leftSum, rightSum + 1, leftMax);
            cur.deleteCharAt(cur.length() - 1);
        }
    }

    // 23. 合并K个升序链表
    public ListNode mergeKLists(ListNode[] lists) {
        ListNode ans = new ListNode(0);
        Queue<ListNode> queue = new PriorityQueue<>((v1, v2) -> v1.val - v2.val);


        //将所有链表的头结点 加入到 优先队列中
        for (int i = 0; i < lists.length; i++) {
            if (lists[i] != null) queue.offer(lists[i]);
        }
        ListNode cur = ans;
        while (!queue.isEmpty()) {
            ListNode poll = queue.poll();
            cur.next = poll;
            cur = cur.next;
            if (poll.next != null) queue.offer(poll.next);
        }
        return ans.next;

    }

    // 31. 下一个排列
    public void nextPermutation(int[] nums) {
        int len = nums.length;
        int right = 0, left = 0;
        for (right = len - 1; right >= 1; right--) {
            if (nums[right - 1] < nums[right]) {
                left = right - 1;
                // 在 [right,len-1]中找到一个比 nums[left] 大的数,进行交换
                for (int i = len - 1; i >= right; i--) {
                    if (nums[i] > nums[left]) {
                        int temp = nums[i];
                        nums[i] = nums[left];
                        nums[left] = temp;
                        break;
                    }
                }
                // 将后面部分排序
                Arrays.sort(nums, right, len);
                break;
            }

        }

        // 特殊情况: 数组原本是逆序排列,那么下一个序列应该为正序的数组
        if (right == 0) Arrays.sort(nums);
    }

    // 32. 最长有效括号 方法一 : 常规解法
    public int longestValidParentheses1(String s) {
        int leftSum = 0;
        int rightSum = 0;
        int maxLen = 0;
        char[] arr = s.toCharArray();
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == '(') leftSum++;
            else rightSum++;
            if (leftSum == rightSum) {
                maxLen = Math.max(maxLen, leftSum + rightSum);
            }
            if (rightSum > leftSum) {
                leftSum = 0;
                rightSum = 0;
            }
        }
        leftSum = rightSum = 0;
        for (int i = arr.length - 1; i >= 0; i--) {
            if (arr[i] == '(') leftSum++;
            else rightSum++;
            if (leftSum == rightSum) {
                maxLen = Math.max(maxLen, leftSum + rightSum);
            }
            if (rightSum < leftSum) {
                leftSum = 0;
                rightSum = 0;
            }
        }
        return maxLen;
    }

    // 32. 最长有效括号 方法二 : 动态规划
    public int longestValidParentheses2(String s) {
        //dp[i] 表示 [0,i]区间里以第i个字符为结尾的有效括号数量
        int dp[] = new int[s.length()];
        Arrays.fill(dp, 0);

        char[] chars = s.toCharArray();

        int maxLen = 0;
        //i从第2个开始,因为有效括号必须为偶数
        for (int R = 1; R < chars.length; R++) {
            if (chars[R] == ')') {
                int L = R - dp[R - 1] - 1;
                if (L >= 0 && chars[L] == '(') dp[R] = 2 + ((L - 1 >= 0) ? dp[L - 1] : 0) + dp[R - 1];

            }
            maxLen = Math.max(maxLen, dp[R]);
        }
        return maxLen;
    }

    // 33. 在旋转排序数组中搜索
    public int search(int[] nums, int target) {
        int len = nums.length;
        if (len == 0) return -1;
        if (len == 1) return nums[0] == target ? 0 : -1;

        int l = 0, r = len - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (nums[mid] == target) return mid;
            if (nums[0] < nums[mid]) { // 左边有序
                if (nums[l] < target && target < nums[mid]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else {
                if (nums[mid] < target && target < nums[r]) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }

        }
        return -1;
    }

    // 34. 在排序数组中查找元素的第一个和最后一个位置
    public int[] searchRange(int[] nums, int target) {
        int n = nums.length;
        if (n == 0) {
            return new int[]{-1, -1};
        }
        int begin = 0;
        int end = 0;
        //查找左边界 begin 大于target -1 的第一个位置
        begin = binarySearch(nums, target - 1);

        //查找右边界 begin 大于target  的第一个位置

        end = binarySearch(nums, target);

        if (nums[begin] == target) return new int[]{begin, end - 1};
        else return new int[]{-1, -1};
    }

    // 寻找大于target 的第一个的位置
    private int binarySearch(int[] nums, int target) {
        int l = 0;
        int r = nums.length - 1;
        int ans = 0;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            //在这里确定 第一个位置的下标
            if (target < nums[mid]) {
                ans = mid;
                r = mid - 1;
            } else {
                l = mid + 1;
            }


        }
        return ans;
    }

    // 39. 组合总和
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> ans = new ArrayList<>();


        fun(candidates, target, 0, new ArrayList<>(), ans);

        return ans;
    }

    private void fun(int[] arr, int rest, int index, List<Integer> temp, List<List<Integer>> ans) {
        if (index == arr.length) return;
        if (rest == 0) {
            ans.add(temp);
            return;
        }
        fun(arr, rest, index + 1, temp, ans);
        if (arr[index] <= rest) {
            temp.add(arr[index]);
            fun(arr, rest - arr[index], index + 1, temp, ans);
            temp.remove(temp.size() - 1);//回溯算法的精髓所在

        }
    }

    // 42. 接雨水
    public int trap(int[] height) {
        int len = height.length;

        int leftMax[] = new int[len];
        leftMax[0] = height[0];
        int rightMax[] = new int[len];
        rightMax[len - 1] = height[len - 1];
        //[0,i] 最大的值
        for (int i = 1; i < len; i++) {
            leftMax[i] = Math.max(height[i], leftMax[i - 1]);
        }
        //[i,len-1]最大值
        for (int i = len - 2; i >= 0; i--) {
            rightMax[i] = Math.max(height[i], rightMax[i + 1]);
        }

        int ans = 0;
        for (int i = 0; i < len; i++) {
            ans += (Math.min(leftMax[i], rightMax[i]) - height[i]) * 1;

        }
        return ans;


    }

    // 46. 全排列
    public List<List<Integer>> permute(int[] nums) {
        int len = nums.length;
        if (len == 0) return null;
        if (len == 1) return new ArrayList<>(new ArrayList(Arrays.asList(nums)));

        List<List<Integer>> ans = new ArrayList<>();
        fun(ans, new ArrayList<>(), nums);
        return ans;
    }

    private void fun(List<List<Integer>> ans, ArrayList<Integer> temp, int[] nums) {
        if (temp.size() == nums.length) {
            //这里注意将新的 ArrayList对象放入 ans中
            ans.add(new ArrayList<>(temp));
//            ans.add(temp);
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            if (!temp.contains(nums[i])) {
                temp.add(nums[i]);
                fun(ans, temp, nums);
                temp.remove(temp.size() - 1);
            }
        }

    }

    //1124. 表现良好的最长时间段
    public int longestWPI(int[] hours) {
        int n = hours.length;
        int[] s = new int[n + 1];
        Stack<Integer> stk = new Stack<>();
        stk.push(0);
        for (int i = 1; i <= n; i++) {
            s[i] = s[i - 1] + (hours[i - 1] > 8 ? 1 : -1);
            if (s[i] < s[stk.peek()]) {
                stk.push(i);
            }
        }

        int res = 0;
        for (int r = n; r >= 1; r--) {
            while (!stk.isEmpty() && s[stk.peek()] < s[r]) {
                res = Math.max(res, r - stk.pop());
            }
        }
        return res;


    }

    // 238. 除自身以外数组的乘积
    public int[] productExceptSelf(int[] nums) {
        int len = nums.length;

        int[] prefixResult = new int[len];
        int[] lastfixResult = new int[len];

        prefixResult[0] = nums[0];

        for (int i = 1; i <= len - 1; i++) {
            prefixResult[i] = prefixResult[i - 1] * nums[i];
        }

        lastfixResult[len - 1] = nums[len - 1];

        for (int i = len - 2; i >= 0; i--) {
            lastfixResult[i] = lastfixResult[i + 1] * nums[i];
        }
        int[] ans = new int[len];
        ans[0] = lastfixResult[1];
        ans[len - 1] = prefixResult[len - 2];
        for (int i = 1; i <= len - 2; i++) {
            ans[i] = prefixResult[i - 1] * lastfixResult[i + 1];
        }
        return ans;
    }

    // 48. 旋转图像
    public void rotate(int[][] matrix) {
        int colLen = matrix.length;
        int rowLen = matrix.length;
        //上下交换
        for (int i = 0; i < rowLen / 2; i++) {
            for (int j = 0; j < colLen; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[colLen - 1 - i][j];
                matrix[colLen - 1 - i][j] = temp;
            }

        }

        // 主对角线交换
        for (int i = 1; i < colLen; i++) {
            int L = 0;
            int R = i;
            while (L < colLen && R < colLen) {

                int temp = matrix[L][R];
                matrix[L][R] = matrix[R][L];
                matrix[R][L] = temp;
                L++;
                R++;
            }
        }

    }

    //49. 字母异位词分组
    public List<List<String>> groupAnagrams(String[] strs) {
        if (strs == null || strs.length == 0) return null;
        List<List<String>> ans = new ArrayList<>();
        Map<String, List<String>> map = new HashMap<>();

        for (String str : strs) {
            char[] key = str.toCharArray();
            Arrays.sort(key);
            List<String> temp = map.getOrDefault(String.valueOf(key), new ArrayList<>());
            temp.add(str);
            map.put(String.valueOf(key), temp);
        }
        return new ArrayList<>(map.values());
    }

    //53. 最大子数组和
    public int maxSubArray(int[] nums) {

        int len = nums.length;

        if (len == 0) return -1;
        //dp[i]表示[0,i]包含nums【i】的最大子数组和
        int[] dp = new int[len];
        dp[0] = nums[0];
        int max = nums[0];
        for (int i = 1; i < len; i++) {
            dp[i] = Math.max(nums[i], dp[i - 1] + nums[i]);
            max = Math.max(dp[i], max);
        }
        return max;

    }

    // 55. 跳跃游戏
    public boolean canJump(int[] nums) {
        int len = nums.length;
        if (len == 0 || len == 1) return true;
        //记录当前小人能走的最大位置的下标，初始为nums[0]
        int maxIndex = nums[0];
        for (int i = 1; i < len; i++) {
            if (maxIndex >= len - 1) return true;
            // 小人无法到达i处
            if (i > maxIndex) return false;
            // 小人可以到达i处，并且在i处向前走的距离更远了，更新maxIndex
            if (i + nums[i] > maxIndex) {
                maxIndex = i + nums[i];
            }
        }

        return false;
    }

    // 56. 合并区间
    public int[][] merge(int[][] intervals) {
        int rowLen = intervals.length;
        Arrays.sort(intervals, (v1, v2) -> v1[0] - v2[0]);
        int[][] ans = new int[rowLen][2];
        int index = -1;


        for (int i = 0; i < rowLen; i++) {
            int L = intervals[i][0]; // 左边界
            int R = intervals[i][1];  //右边界
            if (index == -1 || L > ans[index][1]) ans[++index] = intervals[i];
            else ans[index][1] = Math.max(ans[index][1], R);

        }


        return Arrays.copyOf(ans, index + 1);

    }

    //62. 不同路径
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];

        for (int i = 0; i < n; i++)
            dp[0][i] = 1;
        for (int j = 0; j < m; j++)
            dp[j][0] = 1;

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] += (dp[i][j - 1] + dp[i - 1][j]);
            }
        }

        return dp[m - 1][n - 1];
    }

    //64. 最小路径和
    public int minPathSum(int[][] grid) {
        int colLen = grid.length;
        int rowLen = grid[0].length;

        int[][] dp = new int[colLen][rowLen];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < colLen; i++)
            dp[0][i] = grid[0][i] + dp[0][i - 1];
        for (int i = 1; i < rowLen; i++)
            dp[i][0] = grid[i][0] + dp[i - 1][0];

        for (int i = 1; i < rowLen; i++)
            for (int j = 1; j < colLen; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }

        return dp[rowLen - 1][colLen - 1];

    }

    // 70. 爬楼梯
    public int climbStairs(int n) {

        int dp[] = new int[n + 1];
        dp[1] = 1;
        dp[0] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];

    }

    // 72. 编辑距离 （Very Hard ）
    int[][] memo;

    public int minDistance(String word1, String word2) {

        int len1 = word1.length();
        int len2 = word2.length();
        memo = new int[len1][len2];
        return fun(word1, word2, len1 - 1, len2 - 1);
    }

    private int fun(String word1, String word2, int index1, int index2) {
        if (index1 == -1 || index2 == -1) {
            return Math.max(index2, index1) + 1;
        }
        if (memo[index1][index2] != 0) {
            return memo[index1][index2];
        }
        if (word1.charAt(index1) == word2.charAt(index2)) {
            memo[index1][index2] = fun(word1, word2, index1 - 1, index2 - 1);
            return memo[index1][index2];
        }


        memo[index1][index2] = 0;
        int temp = Math.min(fun(word1, word2, index1, index2 - 1), fun(word1, word2, index1 - 1, index2 - 1));
        memo[index1][index2] = 1 + Math.min(temp, fun(word1, word2, index1 - 1, index2));
        return memo[index1][index2];
    }

    //75. 颜色分类 (荷兰国旗问题)
    public void sortColors(int[] nums) {
        //[0,L]为 0
        int L = -1;
        //[R,nums.length-1] 都为2
        int R = nums.length;
//       (L,R)都为1
        for (int i = 0; i < R; ) {
            if (nums[i] == 0) {
                swap(nums, i, ++L);
                i++;
            } else if (nums[i] == 2) {
                swap(nums, --R, i);
            } else i++;

        }
        for (int num : nums) {
            System.out.println(num);
        }

    }

    private void swap(int[] nums, int a, int b) {
        int temp = nums[a];
        nums[a] = nums[b];
        nums[b] = temp;
    }

    //78. 子集 (返回一个数组所有子集)
    public List<List<Integer>> subsets(int[] nums) {
        if (nums == null || nums.length == 0) return null;

        List<List<Integer>> ans = new ArrayList<>();

        subesetsBack(ans, 0, nums, new ArrayList<>());
        return ans;
    }

    private void subesetsBack(List<List<Integer>> ans, int index, int[] nums, List<Integer> temp) {
        if (index == nums.length) {
            ans.add(new ArrayList<>(temp));
            return;
        }
        //不选择该元素
        subesetsBack(ans, index + 1, nums, temp);

        //选择该元素

        temp.add(nums[index]);
        subesetsBack(ans, index + 1, nums, temp);
        temp.remove(temp.size() - 1);
    }

    //78. 子集 (返回一个数组所有子集) 方法二 : 非递归
    public List<List<Integer>> subsetsWay2(int[] nums) {
        if (nums == null || nums.length == 0) return null;

        List<List<Integer>> ans = new ArrayList<>();
        ans.add(new ArrayList<>());

        for (int num : nums) {
            int size = ans.size();
            for (int i = 0; i < size; i++) {
                //将ans中每一个集合末尾加入num形成新的集合
                List<Integer> integers = ans.get(i);
                ArrayList<Integer> newList = new ArrayList<>(integers);
                newList.add(num);
                //收集这些新的集合
                ans.add(newList);
            }
        }
        return ans;
    }

    // 79. 单词搜索

    boolean visit[][];

    public boolean exist(char[][] board, String word) {
        if (word.length() == 0 || board.length == 0) return false;
        visit = new boolean[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                boolean flag = fun(board, i, j, 0, word);
                if (flag) return true;
            }
        }
        return false;
    }

    private boolean fun(char[][] board, int i, int j, int index, String word) {
        if (index == word.length()) {
            return true;
        }
        if (board[i][j] != word.charAt(index)) return false;


        visit[i][j] = true;
        //往上走
        if (i - 1 >= 0 && visit[i - 1][j] == false) {

            boolean flag = fun(board, i - 1, j, index + 1, word);
            if (flag) return true;

        }

        //往左走
        if (j - 1 >= 0 && visit[i][j - 1] == false) {

            boolean flag = fun(board, i, j - 1, index + 1, word);
            if (flag) return true;

        }
        //往下走
        if (i + 1 < board.length && visit[i + 1][j] == false) {

            boolean flag = fun(board, i + 1, j, index + 1, word);
            if (flag) return true;

        }
        if (j + 1 < board[0].length && visit[i][j + 1] == false) {
            //往右走

            boolean flag = fun(board, i, j + 1, index + 1, word);
            if (flag) return true;
        }
        visit[i][j] = false;
        return false;

    }

    // 94. 二叉树的中序遍历
    public List<Integer> inorderTraversal(TreeNode root) {

        List<Integer> ans = new ArrayList();
        inorder(root, ans);
        return ans;
    }

    private void inorder(TreeNode node, List<Integer> ans) {
        if (node != null) {
            inorder(node.left, ans);
            ans.add(node.val);
            inorder(node.left, ans);
        }
    }

    // 96. 不同的二叉搜索树
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
        return dp[n];
    }

    // 98. 验证二叉搜索树
    long preValue = Long.MIN_VALUE;
    public boolean flag = true;

    public boolean isValidBST(TreeNode root) {
        if (root == null) return true;
        fun(root);
        return flag;


    }

    private void fun(TreeNode node) {

        if (node != null) {
            fun(node.left);

            if (node.val <= preValue) {
                flag = false;

            }
            preValue = node.val;
            fun(node.right);
        }
    }

    // 101. 对称二叉树
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return fun(root.left, root.right);
    }

    private boolean fun(TreeNode L, TreeNode R) {


        if (L == null && R == null) return true;
        if (L == null || R == null || L.val != R.val) return false;

        return fun(L.left, R.right) && fun(L.right, R.left);

    }

    // 102. 二叉树的层序遍历
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>();
        if (root == null) return null;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            List<Integer> temp = new ArrayList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {

                TreeNode node = queue.poll();

                temp.add(node.val);

                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
            }
            ans.add(temp);
        }
        return ans;
    }

    // 104. 二叉树的最大深度
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return fun2(root);

    }

    private int fun2(TreeNode node) {
        if (node == null) return 0;
        int left = fun2(node.left);
        int right = fun2(node.right);
        return 1 + Math.max(left, right);
    }

    // 105. 从前序与中序遍历序列构造二叉树
    private Map<Integer, Integer> inorderMap;

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        inorderMap = new HashMap();
        int N = inorder.length;
        for (int i = 0; i < inorder.length; i++) {
            inorderMap.put(inorder[i], i);
        }

        return fun(preorder, 0, N - 1, 0, N - 1);
    }

    private TreeNode fun(int[] preorder, int preLeft, int preRight, int inLeft, int inRight) {
        if (preLeft > preRight) return null;

        int rootval = preorder[preLeft];
        TreeNode root = new TreeNode(rootval);

        int rootIndex = inorderMap.get(preorder[preLeft]);
        int leftSum = rootIndex - inLeft;

        //构建左子树
        root.left = fun(preorder, preLeft + 1, preLeft + leftSum, inLeft, rootIndex - 1);
        //构建右子树
        root.right = fun(preorder, preLeft + leftSum + 1, preRight, rootIndex + 1, inRight);
        return root;
    }

    // 114.二叉树展开为链表
    public void flatten(TreeNode root) {
        if (root == null) return;

        TreeNode cur = root;
        while (cur != null) {
            if (cur.left == null) cur = cur.right;

            TreeNode left = cur.left;
            TreeNode rightmostNode = left;
            while (rightmostNode.right != null) {
                rightmostNode = rightmostNode.right;
            }


            rightmostNode.right = cur.right;

            cur.right = left;
            cur.left = null;
            cur = cur.right;
        }

    }

    // 121. 买卖股票的最佳时机
    public int maxProfit(int[] prices) {

        int len = prices.length;
        int minBuyPrice = Integer.MAX_VALUE;
        int max = 0;

        for (int i = 0; i < len; i++) {
            if (prices[i] < minBuyPrice) {
                minBuyPrice = prices[i];
            } else if (prices[i] - minBuyPrice > max) {
                max = prices[i] - minBuyPrice;
            }
        }

        return max;
    }

    // 124. 二叉树中的最大路径和
    int maxSum = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        fun5(root);
        return maxSum;
    }

    class Info {
        public int gain;

        public Info(int a) {
            if (a > 0) gain = a;
            else gain = 0;
        }
    }

    public Info fun5(TreeNode node) {
        if (node == null) return new Info(0);

        Info left = fun5(node.left);
        Info right = fun5(node.right);
        //以这个节点为根
        int sum = node.val + left.gain + right.gain;
        //不需要
        maxSum = Math.max(sum, maxSum);
        //不以这个为根
        return new Info(node.val + Math.max(left.gain, right.gain));
    }

    // 128. 最长连续序列
    public int longestConsecutinve(int[] nums) {
        Set<Integer> num_set = new HashSet<>();
        for (int num : nums) {
            num_set.add(num);
        }
        int max = 1;
        for (int num : nums) {
            if (num_set.contains(num - 1)) continue;
            int temp = num;
            int len = 1;
            while (num_set.contains(temp + 1)) {
                temp++;
                len++;
            }
            max = Math.max(len, max);
        }
        return max;
    }

    // 136. 只出现一次的数字
    public int singleNumber(int[] nums) {
        int e = 0;
        for (int num : nums) {
            e ^= num;
        }
        return e;

    }

    // 139. 单词拆分  动态规划
    public boolean wordBreak(String s, List<String> wordDict) {
        boolean[] dp = new boolean[s.length() + 1];

        Arrays.fill(dp, false);
        dp[0] = true;

        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (wordDict.contains(s.substring(j, i)) && dp[j]) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];


    }

    // 141. 环形链表 1
    public boolean hasCycle(ListNode head) {

        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) return true;
        }
        return false;
    }

    // 142. 环形链表 II
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;

        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                fast = head;
                while (fast != slow) {
                    slow = slow.next;
                    fast = fast.next.next;
                }
                return fast;
            }

        }

// 无环
        return null;

    }

    // 146. LRU缓存
    // 双向链表节点类
    class Node {
        public int key;
        public int value;
        public Node last;
        public Node next;

        public Node() {

        }

        public Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }

    // 双向链表类
    class DoubleLinkedList {
        private Node head;
        private Node tail;


        public DoubleLinkedList() {
            this.head = new Node();
            this.tail = new Node();
            head.next = tail;
            tail.last = head;
        }

        /**
         * 在尾部添加节点  尾部频率高
         */
        public void addNode(Node node) {
            if (node == null) return;
            node.next = tail;
            node.last = tail.last;
            tail.last.next = node;
            tail.last = node;

        }

        /**
         * 将尾部节点移到头结点,尾部节点 频率最高
         */
        public void addtoTail(Node node) {
            if (tail == node) return;


            node.next = tail;
            node.last = tail.last;
            tail.last = node;
            tail.last.next = node;
        }

        /**
         * 删除最久未使用的节点--头结点
         */
        public Node removeHead() {
            Node ans = head.next;
            ans.next.last = ans.last;
            ans.last.next = ans.next;

            return ans;
        }
    }

    class LRUCache {


        private HashMap<Integer, Node> keyMap;
        private DoubleLinkedList nodeList;
        int capacity;

        public LRUCache(int capacity) {

            keyMap = new HashMap<>();
            nodeList = new DoubleLinkedList();
            this.capacity = capacity;
        }

        public int get(int key) {
            if (keyMap.containsKey(key)) {
                Node ans = keyMap.get(key);
                nodeList.addtoTail(ans);
                return ans.value;
            }
            return -1;
        }

        public void put(int key, int value) {
            if (keyMap.containsKey(key)) {
                Node node = keyMap.get(key);
                nodeList.addtoTail(node);
                node.value = value;
            } else {
                Node node = new Node(key, value);
                keyMap.put(key, node);
                nodeList.addNode(node);
                if (keyMap.size() > capacity) {
                    Node temp = nodeList.removeHead();
                    keyMap.remove(temp.key);

                }
            }
        }
    }

    // 148. 排序链表  分治算法(递归版本)
    public ListNode sortList(ListNode head) {
        return sortList(head, null);
    }

    private ListNode sortList(ListNode l, ListNode r) {
        System.out.println("====");
        System.out.println("当前 l为: " + l.val + ", r为 : " + (r == null ? null : r.val));
        if (l == null) return l;
        if (l.next == r) {
            l.next = null;
            return l;
        }
        ListNode slow = l;
        ListNode fast = l;
        while (fast != r) {
            fast = fast.next;
            slow = slow.next;
            if (fast != r) {
                fast = fast.next;
            }
        }

        ListNode mid = slow;
        System.out.println("mid 为 : " + mid.val);
        ListNode list1 = sortList(l, mid);
        System.out.print("(l,mid)结果为: ");
        Util.printList(list1);
        ListNode list2 = sortList(mid, r);
        System.out.print("(mid,r)结果为: ");
        Util.printList(list2);
        ListNode sorted = merge(list1, list2);
        System.out.print("(l,r)合并后为: ");
        Util.printList(sorted);
        return sorted;

    }

    private ListNode merge(ListNode head1, ListNode head2) {
        ListNode dummyHead = new ListNode(0);
        ListNode temp = dummyHead, temp1 = head1, temp2 = head2;
        while (temp1 != null && temp2 != null) {
            if (temp1.val <= temp2.val) {
                temp.next = temp1;
                temp1 = temp1.next;
            } else {
                temp.next = temp2;
                temp2 = temp2.next;
            }
            temp = temp.next;
        }
        if (temp1 != null) {
            temp.next = temp1;
        } else if (temp2 != null) {
            temp.next = temp2;
        }
        return dummyHead.next;

    }

    // 152 乘积最大的数组
    public int maxProduct(int[] nums) {
        int max = 1;
        int min = 1;
        int ans = Integer.MIN_VALUE;
        for (int num : nums) {
            if (num < 0) {
                min = min ^ max;
                max = min ^ max;
                min = min ^ max;
            }
            max = Math.max(num, num * max);
            min = Math.min(num, num * min);
            ans = Math.max(max, ans);
        }
        return ans;

    }

    //155. 最小栈
    class MinStack {
        Stack<Integer> dataStack;
        Stack<Integer> minStack;

        public MinStack() {
            dataStack = new Stack();
            minStack = new Stack();
        }

        public void push(int val) {
            dataStack.push(val);
            if (minStack.isEmpty()) {
                minStack.push(val);
            } else minStack.push(Math.min(minStack.peek(), val));
        }

        public void pop() {
            dataStack.pop();
            minStack.pop();
        }

        public int top() {
            return dataStack.peek();
        }

        public int getMin() {
            return minStack.peek();
        }
    }

    // 160. 相交链表
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        ListNode pA = headA, pB = headB;
        while (pA != pB) {
            pA = pA == null ? headB : pA.next;
            pB = pB == null ? headA : pB.next;
        }
        return pA;

    }

    // 169. 多数元素
    // 思想: 众数到最后一定比其它数多,每次遍历记录某一个数X的出现次数,遇到不相同的就-1,直到0,更换其它众数Y
    public int majorityElement(int[] nums) {
        int ans = nums[0];
        int count = 1;
        for (int num : nums) {
            if (num == ans) {
                count++;
            } else if (count == 0) {
                ans = num;
                count = 1;
            } else count--;
        }
        return ans;
    }

    // 198. 打家劫舍  空间复杂度O(n),时间复杂度O(n)
    public int rob(int[] nums) {
        int len = nums.length;
        //[0,i)中最大的金额
        int[] dp = new int[len + 1];
        dp[0] = 0;
        dp[1] = nums[0];

        for (int i = 1; i < len; i++) {
            dp[i + 1] = Math.max(dp[i - 1] + nums[i], dp[i]);
        }


        return dp[len];
    }

    // 198. 打家劫舍 法二 空间复杂度O(1),时间复杂度O(n)
    public int rob2(int[] nums) {
        int pre_2 = 0;
        int pre_1 = 0;
        for (int num : nums) {
            int temp = Math.max(num + pre_2, pre_1);
            pre_2 = pre_1;
            pre_1 = temp;

        }
        return pre_1;
    }

    // 200. 岛屿数量 DFS 时间复杂度：O(MN)，空间复杂度：O(MN)
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) return 0;
        int rowLen = grid.length;
        int cowLen = grid[0].length;
        int ans = 0;
        for (int i = 0; i < rowLen; ++i) {
            for (int j = 0; j < cowLen; ++j) {
                if (grid[i][j] == '1') {
                    ans++;
                    dfs(grid, i, j);

                }
            }
        }
        return ans;
    }

    private void dfs(char[][] grid, int i, int j) {
        int rowLen = grid.length;
        int cowLen = grid[0].length;
        if (i < 0 || j < 0 || i >= rowLen || j > cowLen || grid[i][j] == '0') {
            return;
        }


        grid[i][j] = '0';

        //向四个方向扩散
        dfs(grid, i - 1, j);
        dfs(grid, i + 1, j);
        dfs(grid, i, j - 1);
        dfs(grid, i, j + 1);

    }

    // 200. 岛屿数量 BFS 时间复杂度：O(MN)，空间复杂度：O(1)
    public int numIslands2(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }

        int rowLen = grid.length;
        int colLen = grid[0].length;
        int num_islands = 0;

        for (int i = 0; i < rowLen; ++i) {
            for (int j = 0; j < colLen; ++j) {
                if (grid[i][j] == '1') {
                    ++num_islands;
                    grid[i][j] = '0';
                    Queue<Integer> neighbors = new ArrayDeque<>();
                    neighbors.add(i * colLen + j);
                    while (!neighbors.isEmpty()) {
                        int id = neighbors.remove();
                        int row = id / colLen;
                        int col = id % colLen;
                        if (row - 1 >= 0 && grid[row - 1][col] == '1') {
                            neighbors.add((row - 1) * colLen + col);
                            grid[row - 1][col] = '0';
                        }
                        if (row + 1 < rowLen && grid[row + 1][col] == '1') {
                            neighbors.add((row + 1) * colLen + col);
                            grid[row + 1][col] = '0';
                        }
                        if (col - 1 >= 0 && grid[row][col - 1] == '1') {
                            neighbors.add(row * colLen + col - 1);
                            grid[row][col - 1] = '0';
                        }
                        if (col + 1 < colLen && grid[row][col + 1] == '1') {
                            neighbors.add(row * colLen + col + 1);
                            grid[row][col + 1] = '0';
                        }
                    }
                }
            }
        }

        return num_islands;
    }

    // 200. 岛屿数量 并查集 时间复杂度：O(MN)，空间复杂度：O(MN)
    public int numIslands3(char[][] grid) {
        int rowLen = grid.length;
        int colLen = grid[0].length;
        UnionFind uf = new UnionFind(grid);
        for (int r = 0; r < rowLen; ++r) {
            for (int c = 0; c < colLen; ++c) {
                if (grid[r][c] == '1') {
                    grid[r][c] = '0';
                    if (r - 1 >= 0 && grid[r - 1][c] == '1') {
                        uf.unionElements(r * colLen + c, (r - 1) * colLen + c);
                    }
                    if (r + 1 < rowLen && grid[r + 1][c] == '1') {
                        uf.unionElements(r * colLen + c, (r + 1) * colLen + c);
                    }
                    if (c - 1 >= 0 && grid[r][c - 1] == '1') {
                        uf.unionElements(r * colLen + c, r * colLen + c - 1);
                    }
                    if (c + 1 < colLen && grid[r][c + 1] == '1') {
                        uf.unionElements(r * colLen + c, r * colLen + c + 1);
                    }
                }
            }
        }
        return uf.getSize();

    }

    //206. 反转链表
    public ListNode reverseList(ListNode head) {
        ListNode cur = head;
        ListNode last = null;
        while (cur != null) {
            ListNode nextNode = cur.next;
            cur.next = last;
            last = cur;
            cur = nextNode;
        }
        return last;
    }

    //207. 课程表
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        return false;
    }
}
