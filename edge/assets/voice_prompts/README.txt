Put pre-recorded Chinese WAV prompt files here.

Required demo prompt files:
01. system_ready.wav - 系统就绪
02. inventory_reset.wav - 库存已清空
03. inventory_initialized.wav - 库存已按当前画面初始化
04. no_stable_snapshot.wav - 当前没有稳定识别结果
05. camera_quality_bad.wav - 当前画面质量较差，请调整光照或遮挡
06. label_saved.wav - 标签录入成功，已记录保质期和营养信息
07. label_failed.wav - 标签识别失败，请重新对准包装标签
08. reminder_saved.wav - 已设置食用提醒
09. assistant_ready.wav - 智能助手已就绪
10. apple_add.wav - 苹果已入库
11. apple_remove.wav - 苹果已取出
12. banana_add.wav - 香蕉已入库
13. banana_remove.wav - 香蕉已取出
14. milk_add.wav - 盒装牛奶已入库
15. milk_remove.wav - 盒装牛奶已取出
16. coke_add.wav - 罐装饮料已入库
17. coke_remove.wav - 罐装饮料已取出
18. egg_add.wav - 鸡蛋已入库
19. egg_remove.wav - 鸡蛋已取出
20. blueberry_add.wav - 蓝莓已入库
21. blueberry_half.wav - 检测到半盒蓝莓，库存已更新
22. blueberry_remove.wav - 蓝莓已取出
23. multi_add.wav - 多种食材已入库
24. multi_remove.wav - 多种食材已取出
25. inventory_empty.wav - 当前冰箱库存为空
26. inventory_updated.wav - 库存已更新
27. eat_soon.wav - 建议尽快食用临期食材
28. milk_expire_reminder.wav - 牛奶临期提醒已生成
29. coke_nutrition_saved.wav - 饮料营养信息已保存
30. milk_nutrition_saved.wav - 牛奶营养信息已保存

The demo plays WAV files first when a mapped prompt is spoken.
If a WAV file is missing, it falls back to the configured TTS command or espeak-ng.
Keep filenames lowercase and exactly matching fridge_demo_config.json.
