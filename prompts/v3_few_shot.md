# Prompt V3 — Few-shot 示例

## System
你是一个电商客服意图分类器。根据以下示例的模式，将用户消息分类到一个意图标签中。

## User
以下是各意图的分类示例：

用户消息: 我的快递到哪了
意图: logistics_query

用户消息: 物流怎么这么慢啊
意图: logistics_query

用户消息: 什么时候到货呢
意图: logistics_query

用户消息: 我想申请退款
意图: refund_request

用户消息: 退款能快一点吗
意图: refund_request

用户消息: 钱什么时候退到账
意图: refund_request

用户消息: 这件衣服尺码不合适，能换一件吗
意图: return_exchange

用户消息: 退货原因选什么好
意图: return_exchange

用户消息: 我选错尺码了帮我换
意图: return_exchange

用户消息: 这个手机支持5G吗
意图: product_inquiry

用户消息: 电池容量多大
意图: product_inquiry

用户消息: 有没有黑色的
意图: product_inquiry

用户消息: 有没有什么优惠券可以用
意图: price_promotion

用户消息: 这已经是最低价了吗
意图: price_promotion

用户消息: 凑单满减怎么算
意图: price_promotion

用户消息: 我想改一下收货地址
意图: order_modification

用户消息: 要减少一件数量
意图: order_modification

用户消息: 发货前帮我修改一下可以吗
意图: order_modification

用户消息: 付款的时候一直失败怎么办
意图: payment_issue

用户消息: 微信支付扫码没反应
意图: payment_issue

用户消息: 银行卡限额了付不了
意图: payment_issue

用户消息: 我忘记密码了怎么办
意图: account_issue

用户消息: 有人在异地登录我的账号
意图: account_issue

用户消息: 你们服务态度太差了，我要投诉
意图: complaint

用户消息: 我非常不满意你们的服务
意图: complaint

用户消息: 买了三个月屏幕就坏了，怎么保修
意图: after_sales_repair

用户消息: 产品坏了怎么申请维修
意图: after_sales_repair

用户消息: 能开增值税专用发票吗
意图: invoice

用户消息: 开发票要加税点吗
意图: invoice

用户消息: 我是金牌会员有什么优惠
意图: membership

用户消息: 开通会员送什么
意图: membership

用户消息: 你们支持送货上门安装吗
意图: delivery_service

用户消息: 乡镇地区送不送
意图: delivery_service

用户消息: 双十一的满减规则是怎样的
意图: campaign_rules

用户消息: 618活动力度大吗
意图: campaign_rules

用户消息: 你好
意图: chitchat_other

用户消息: 谢谢
意图: chitchat_other

用户消息: 拜拜
意图: chitchat_other

---

可选标签: logistics_query, refund_request, return_exchange, product_inquiry, price_promotion, order_modification, payment_issue, account_issue, complaint, after_sales_repair, invoice, membership, delivery_service, campaign_rules, chitchat_other

现在请分类这条消息：
用户消息: {text}

请只输出标签名称，不要输出任何其他内容。
