def get_slope(df, start_q, end_q, idx):
    start_size = df.loc[(df['quality'] == start_q), 'size'].iloc[0]
    end_size = df.loc[(df['quality'] == end_q), 'size'].iloc[0]
    if idx == 0:
        start = df.loc[(df['quality'] == start_q), 'psnr'].iloc[0]
        end = df.loc[(df['quality'] == end_q), 'psnr'].iloc[0]
    elif idx == 1:
        start = df.loc[(df['quality'] == start_q), 'l1'].iloc[0]
        end = df.loc[(df['quality'] == end_q), 'l1'].iloc[0]    
    elif idx == 2:
        start = df.loc[(df['quality'] == start_q), 'l2'].iloc[0]
        end = df.loc[(df['quality'] == end_q), 'l2'].iloc[0]
    
    if end_size - start_size <= 0:
        slope = 999999
    else:
        slope = (end - start) / (end_size - start_size)

    return slope