def collate_text_only(batch):
    # batch = list[ (prompt, goal_id, goal_lbl, frame_lbl) ]
    texts, gid, glbl, flbl = zip(*batch)
    return list(texts), torch.tensor(gid), \
           torch.tensor(glbl), torch.tensor(flbl)
