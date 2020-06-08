## Pandas相关使用方法汇总

1. 合并数据框

   ```python
   merge_df = pd.merge(left=A_df, right=B_df, how='left', left_on=['A', 'B'], right_on=['A', 'B'])
   ```

2. 去除重复行

   ```Python
   unique_res_df = res_df.drop_duplicates(subset=['A', 'B'], keep='first', inplace=False)
   ```

3. 所有行多列元素调用一个方法

   ```python
   res_df['res'] = res_df.apply(lambda x: func(x['A'], x['B']), axis=1)
   ```

4. 透视表（统计某两个指标的对应关系）

   ```python
   table_df = pd.pivot_table(df, index=['A', 'B'], values=['C'], aggfunc=len)
   ```

5. 挑选数据框中某列含有nan值的所有行

   ```python
   nan_df = all_df[ps.isnull(all_df['A'])]
   ```

6. 多列操作得到多列的值

   ```python
   res_df['C'], res_df['D'] = zip(*res_df.apply(lambda x: func(x['A'], x['B']), axis=1))
   ```

7. 