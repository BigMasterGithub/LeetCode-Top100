import assist.ListNode;
import assist.TreeNode;
import cn.hutool.core.text.csv.CsvUtil;
import sort.QuickSort;
import utils.Util;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Queue;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/2/2 18:33
 **/
public class Tests {


    public static void main(String[] args) {
        Solution solution = new Solution();

       /* int[] ans = solution.twoSum(new int[]{12, 2, 4, 5, 6, 1, 5}, 18);
        System.out.println(ans[0]+","+ans[1]);
        ListNode l1  = Util.createList(3);
        ListNode l2  = Util.createList(4);
        Util.printList(l1);
        Util.printList(l2);

        ListNode ans2 = solution.addTwoNumbers(l1, l2);
        Util.printList(ans2);

        int ans3 = solution.lengthOfLongestSubstring("aaabcdefggag");
        System.out.println(ans3);


        double ans = solution.findMedianSortedArrays(new int[]{1}, new int[]{5,6});
        System.out.println(ans);


        String ans =  solution.longestPalindrome("abcbaggga");
        System.out.println(ans);


        int ans = solution.maxArea(new int[]{1, 8, 6, 2, 5, 4, 8, 3, 7});
        System.out.println(ans);


        List<List<Integer>> ans = solution.threeSum(new int[]{-1, 0, 1, 2, -1, -4});
        for (List temp : ans){
            System.out.println(temp);
        }

        List<String> ans = solution.letterCombinations("23");
        for (String temp : ans){
            System.out.println(temp);
        }

        boolean ans = solution.isValid("{}{}()()[({})]");
        System.out.println(ans);

        ListNode ans = solution.mergeTwoLists(Util.createSortedList(5), Util.createSortedList(5));
        Util.printList(ans);

        List<String> ans = solution.generateParenthesis(3);
        for(String cur :ans){
            System.out.println(cur);
        }

        ListNode list1 = Util.createSortedList(5);
        ListNode list2 = Util.createSortedList(6);
        ListNode list3 = Util.createSortedList(7);
        ListNode[] lists = new ListNode[]{list1, list2, list3};
        ListNode ans = solution.mergeKLists(lists);
        Util.printList(ans);


        solution.nextPermutation(new int[]{3, 4, 9, 8, 7, 6});


        int ans = solution.longestValidParentheses1("()()()()))(()()(())()");

        System.out.println(ans);

        int[] ans = solution.searchRange(new int[]{6, 7, 7, 7, 7, 8, 9, 10}, 7);
        System.out.println(ans[0] + " , " + ans[1]);

        int ans = solution.trap(new int[]{4,2,0,3,2,5});
        System.out.println(ans);

        for (List<Integer> integers : solution.permute(new int[]{1, 3, 2})) {
            System.out.println(integers);      }




        for (int i1 : solution.productExceptSelf(new int[]{1, 2, 3, 4})) {
            System.out.println(i1);
        }

        List<List<String>> lists = solution.groupAnagrams(new String[]{"eat", "tea", "tan", "ate", "nat", "bat"});
        for (List<String> list : lists) {
            System.out.println(list);
        }

        int ans = solution.maxSubArray(new int[]{-2, 1, -3, 4, -1, 2, 1, -5, 4});
        System.out.println(ans);

        boolean b = solution.canJump(new int[]{3,2,1,0, 4});
        System.out.println(b);

        solution.sortColors(new int[]{2,0,2,1,1,0});



        boolean exist = solution.exist(new char[][]{{'A', 'B', 'C', 'E'}, {'S', 'F', 'C', 'S'}, {'A', 'D', 'E', 'E'}}, "ABDEEEF");
        System.out.println(exist);

        List<String> dict = new ArrayList<>();
        dict.add("leet");
        dict.add("code");
        boolean applepenapple = solution.wordBreak("leetcode", dict);

        ListNode list = Util.createList(8);
        Util.printList(list);
        solution.sortList(list);


        LRUCache cache = new LRUCache(2);
        cache.put(1, 1);
        cache.put(2, 3);
        cache.get(1);
        cache.put(3, 3);
        System.out.println(cache.get(2));
        cache.put(4, 4);
        cache.get(1);
        cache.get(3);
        cache.get(4);

        int rob = solution.rob(new int[]{9,2,1,8,6,5});
        System.out.println(rob);

        ListNode list1 = Util.createSortedList(5);
        Util.printList(list1);
        solution.reverseList(list1);
        System.out.println("旋转后:"+list1);


        boolean b = solution.canFinish(3, new int[][]{{0, 2}, {1, 2}, {1, 0}});



        int[] nums = new int[]{100, 3, 41, 123, 4, 56, 8, 36, 1};
        System.out.println(Arrays.toString(nums));
        QuickSort q = new QuickSort();
        q.quickSort(nums, 0, 8);
        System.out.println(Arrays.toString(nums));


        int peakElement = solution.findPeakElement(new int[]{1, 3, 2, 1});
        System.out.println(peakElement);



        solution.maxSlidingWindow(new int[]{1, 3, -1, -3, 5, 3, 6, 7}, 3);



        solution.moveZeroes(new int[]{1,0,0,8,6});

        */
        TreeNode root = new TreeNode(1);
        TreeNode node2 = new TreeNode(2);
        TreeNode node3 = new TreeNode(3);
        TreeNode node4= new TreeNode(4);
        TreeNode node5= new TreeNode(5);
        TreeNode node6= new TreeNode(6);

        root.left =node2;
        root.right =node3;


        node3.left=node4;
        node3.right=node5;




    }


}

