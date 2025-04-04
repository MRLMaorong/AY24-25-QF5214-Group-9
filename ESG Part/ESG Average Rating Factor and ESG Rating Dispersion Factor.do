//数据计算过程
//按年对数据进行排序后进行分位数分组，分成一百组
//标准化，进行百分位排名
//计算两两机构配对标准差


ssc install egenmo
rename wind Wind
//分组：xtile命令对数据进行排序后的分位数分组，分成了100组
xtset stock year
bysort year :egen msci_ptile =xtile( msci ),nq(100)
bysort year :egen Wind_ptile =xtile( Wind ),nq(100)
bysort year :egen 华证_ptile =xtile( 华证 ),nq(100)
bysort year :egen 富时罗素_ptile =xtile( 富时罗素 ),nq(100)
bysort year :egen 商道融绿_ptile =xtile( 商道融绿 ),nq(100)
bysort year :egen 盟浪_ptile =xtile( 盟浪 ),nq(100)
//算出归一化百分位排名，统一计量单位，为后续做标准差打基础
gen msci_rank= msci_ptile/100
gen Wind_rank= Wind_ptile /100
gen 华证_rank= 华证_ptile /100
gen 富时罗素_rank= 富时罗素_ptile /100
gen 商道融绿_rank= 商道融绿_ptile /100
gen 盟浪_rank= 盟浪_ptile /100
//rowsd命令，得出两两匹配机构的联合标准差
egen msciwind = rowsd( msci_rank Wind_rank )
egen msci华证 = rowsd( msci_rank 华证_rank )
egen msci富时 = rowsd( msci_rank 富时罗素_rank )
egen msci融绿 = rowsd( msci_rank 商道融绿_rank )
egen msci盟浪 = rowsd( msci_rank 盟浪_rank )
egen wind华证 = rowsd( Wind_rank 华证_rank )
egen wind富时 = rowsd( Wind_rank 富时罗素_rank )
egen wind融绿 = rowsd( Wind_rank 商道融绿_rank )
egen wind盟浪 = rowsd( Wind_rank 盟浪_rank )
egen 华证富时 = rowsd( 华证_rank 富时罗素_rank )
egen 华证融绿 = rowsd( 华证_rank 商道融绿_rank )
egen 华证盟浪 = rowsd( 华证_rank 盟浪_rank )
egen 富时融绿 = rowsd( 富时罗素_rank 商道融绿_rank )
egen 富时盟浪 = rowsd( 富时罗素_rank 盟浪_rank )
egen 融绿盟浪 = rowsd( 商道融绿_rank 盟浪_rank )
//得出两两匹配机构的行均值
egen msciwind1 = rowmean( msci_rank Wind_rank )
egen msci华证1 = rowmean( msci_rank 华证_rank )
egen msci富时1 = rowmean( msci_rank 富时罗素_rank )
egen msci融绿1 = rowmean( msci_rank 商道融绿_rank )
egen msci盟浪1 = rowmean( msci_rank 盟浪_rank )
egen wind华证1 = rowmean( Wind_rank 华证_rank )
egen wind富时1 = rowmean( Wind_rank 富时罗素_rank )
egen wind融绿1 = rowmean( Wind_rank 商道融绿_rank )
egen wind盟浪1 = rowmean( Wind_rank 盟浪_rank )
egen 华证富时1 = rowmean( 华证_rank 富时罗素_rank )
egen 华证融绿1 = rowmean( 华证_rank 商道融绿_rank )
egen 华证盟浪1 = rowmean( 华证_rank 盟浪_rank )
egen 富时融绿1 = rowmean( 富时罗素_rank 商道融绿_rank )
egen 富时盟浪1 = rowmean( 富时罗素_rank 盟浪_rank )
egen 融绿盟浪1 = rowmean( 商道融绿_rank 盟浪_rank )


***计算总体标准差，ESG_uncertainty_all是六家机构的联合标准差，ESG_rank_all是六家机构评级分数的行均值
egen ESG_uncertainty_all = rowsd( msci_rank Wind_rank 华证_rank 富时罗素_rank 商道融绿_rank 盟浪_rank )
egen ESG_rank_all = rowmean( msci_rank Wind_rank 华证_rank 富时罗素_rank 商道融绿_rank 盟浪_rank )
**计算两两标准差的平均数  ESG_rank是两两机构匹配之后行均值求总  ， ESG_uncertainty是两两机构匹配之后标准差再求行均值
egen ESG_uncertainty = rowmean( msciwind msci华证 msci富时 msci融绿 msci盟浪  wind华证 wind富时 wind融绿 wind盟浪 华证富时 华证融绿 华证盟浪  富时融绿 富时盟浪 融绿盟浪 )
egen ESG_rank = rowmean( msciwind1 msci华证1 msci富时1 msci融绿1 msci盟浪1  wind华证1 wind富时1 wind融绿1 wind盟浪1  华证富时1 华证融绿1 华证盟浪1  富时融绿1 富时盟浪1  融绿盟浪1  )





