package utils;

import assist.ListNode;
import cn.hutool.Hutool;
import cn.hutool.core.util.RandomUtil;

import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * @author
 * @description TODO
 * @since 2023/2/2 19:06
 **/
public class Util {
    public static ListNode createList(int length) {
        if (length <= 0) return null;
        ListNode head = new ListNode(RandomUtil.randomInt(1, 9));
        ListNode cur = head;
        for (int i = 1; i < length; i++) {
            ListNode temp = new ListNode(RandomUtil.randomInt(1, 100));
            cur.next = temp;
            cur = cur.next;
        }

        return head;
    }
    public static ListNode createSortedList(int length) {
        if (length <= 0) return null;
        ListNode head = new ListNode(RandomUtil.randomInt(10));
        ListNode cur = head;
        for (int i = 1; i < length; i++) {
            ListNode temp = new ListNode(RandomUtil.randomInt(i*10+1,(i+1)*10));
            cur.next = temp;
            cur = cur.next;
        }

        return head;
    }
    public static void printList(ListNode head) {
        ListNode cur = head;
        while (cur != null) {
            System.out.print(cur.val+" -> ");
            cur = cur.next;
        }
        System.out.println();
    }
}
