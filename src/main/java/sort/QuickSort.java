package sort;

/**
 * @author 张壮
 * @description TODO
 * @since 2023/3/3 20:50
 **/
public class QuickSort {

    
    public void quickSort(int[] nums, int l, int r) {
        if (l >= r) return;
        int p = partition(nums, l, r);
        quickSort(nums, l, p - 1);
        quickSort(nums, p + 1, r);
    }

    public int partition(int[] nums, int L, int R) {
        int j = L;
        int pivot = nums[L];
        for (int i = L + 1; i <= R; i++) {
            if (nums[i] < pivot) {
                j++;
                swap(nums, i, j);
            }

        }
        swap(nums, L, j);
        return j;
    }

    public void swap(int[] nums, int i, int j) {
        int temp =  nums[j];
        nums[j] = nums[i] ;
        nums[i] = temp;
    }
}

