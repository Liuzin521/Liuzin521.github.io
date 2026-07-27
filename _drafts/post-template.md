---
title: '文章标题写在这里（这行显示在页面顶部，中英共用，建议用英文）'
date: 2026-01-01
permalink: /posts/2026/01/your-post-slug/
tags:
  - 标签1
  - 标签2
---

<div class="lang lang-en" markdown="1">

English content goes here. Normal Markdown works inside this block.

</div>

<div class="lang lang-zh" markdown="1">

中文内容写在这里，块内正常写 Markdown 即可。

</div>

<!--
写新博客的方法：
1. 复制这个文件到 _posts/ 目录
2. 文件名改成 "YYYY-MM-DD-英文短标题.md"，例如 2026-08-01-my-second-post.md
3. 改上面 front matter 里的 title / date / permalink / tags
   （permalink 建议和文件名保持一致：/posts/年/月/英文短标题/）
4. 英文写进 lang-en 的 div，中文写进 lang-zh 的 div，页面会自动出现
   "中文 / English" 切换按钮（读者的选择会被记住）
   注意：div 标签和正文之间要留空行，markdown="1" 不能删，否则 Markdown 不渲染
5. 只想写单语言：删掉整段 div 结构直接写正文即可，按钮不会出现
6. 写完 git add + commit + push，网站几分钟后自动更新

放在 _drafts/ 目录里的文件不会发布，可以安心打草稿。
-->
