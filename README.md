# SagaRNN

Inspired by Andrej Karpathy's [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), I was curious to see how well a recurrent neural network (RNN) would work for generating [Icelandic Sagas](https://en.wikipedia.org/wiki/Sagas_of_Icelanders). I created a dataset from an [online collection of the Icelandic sagas](https://www.snerpa.is/net/isl/band.htm) and built a neural network in PyTorch. Of course this model can be trained on any text if wanted so it is not limited to learning just Icelandic Sagas.

## Running the code
To train
```
python3 main.py train
```
To generate text from trained model
```
python3 main.py generate --pretrained_model my_model_name.pt
```
To see additional optional arguments
```
python3 main.py -h
```

## Results
When working on this project, I did not have access to a GPU, but training on the CPU proved to be fast enough that it wasn't needed. The following text is the result from training for 1000 epochs with the default parameters given. To people who do not speak Icelandic this result might look like normal Icelandic text, but to a native Icelander reading this text is likely the closest approximation to what having a stroke feels like. It is very interesting how the model managed to learn some features of Icelandic in such a short time. It has learnt Icelandic names such as "Þórður" and "Ásmundur" and correct capitalization. It has also learned many nouns and even made new nouns that sound plausible. Overall this was a fun side-project to satisfy my curiosity. 



```
Þá mælti: "Fjárinn greiði sama við heim og skal eigi liðan og vilja að gera hvar þeirra ban, að setti stundu til blóðir og var að því gert við þegar í bænum að gera tættina gera af Hrútaferður á mun skips hinn bræðífilla brá: "Svo hafði sækja þá er þeir kallaður hendi frá hljóp maður og sitja hann með þeims en sem það er satt er háskrengur kerling hitta því að svo að mér er byggður fjölmenni við sem þök þar fyrir sár og Þorgeir leiddi til tíðind ok síðar og er Skeggi málar við orðið af vilja hennar þann var fara til Karl á kunni og tóku hann en þeir voru til búinn Hrafnkel þess að hann söllum hundur og Lamlofa og líklega vörð og mælti: "Eigi hefir hann er þér eftir yfir liði gera og gekk hefnskapamann goði spikkju þeir konungs upp uð síðan af þinginum og hestur á öllum allan um sem víkingur þrjó marga með sér þá mikil spjótið. Var maðurt með sér Önundum og var að hann landið og var svo fóru stafnmýrum, en þá allur á með sínu er hann mun þér væri hvar öll lífar, en eg þó er Kári konungi fara fyrir hennarson. Voru fundust þeirra þar að skip það til mót: "Eg hann fór Gísli líklegt."

Síðan kómi var út feng og vel til Njálssyni og er þar slíkt er hann hlut mesti er þú tók sem fyrirlegst veður. Líkaði frændsæll verið eftir lofa, því verða við jarl og bjó bræðrum mjög og sagði: "Það mun vil eg hinn hefir með Ásmundur var kominn hríð til að jarl og hafði þá að þeir sér til lauk svert, er þér er vísur héldu fá að hann komu í hendur til verjast yð fund og varð sinnar. Það er þetta alburðist þú hafi hann hlutum sest fyrir vera en þú var frá gerið vera og gert er eigi kallaður látum svo mikið. Hún spurði seindi en Þorkell var hann svo að eg nú hann og sá er Þorgils fóru þér vilja hann á skip og hafði láta manna og mælti: "Ver hans trúafirðinum stafnt setti þér þetta á bragð við allir og vildi hvar sári elta langi sem höfðingur var eigi gefið víða drengið hjá það tjórið sama þó að þú frændi vestur með fyrir Kormákur gerðinni að hann skalskip og sporinn var bróðir hinn að honum þar fyrir fylgjar. Það allt við fyrir bæjar sem gaf skyldi henni en verði í Vigforð nokkura og sagði það af þig nokkura og minnir að fyrir Þorsteinn en hann gera standa hljóp og sagði hans ef þú ert ætla en því heitir er hann var nær við stórmennt sjá engi undir sveininn sér að hún fóstra þeim kappi þenna sem hann heitir það upp síðan í bléðin og skal hann vilja skaplegt og hæll voru þar á Hólmdi.

Síðan kvaðst er þér ríða maður er fylgjum týstur við nokkuð og höfðu að hann var þar að Bróðir hann en sé er mér líkaði og kvað þeir hafði hann staddur oss og kvaðst nú að hann var traust sem það skilja þykir óvart og lögfa ef það vil að gilja við. Hún en hann kvaðst hún.

Þá ætla vorst undir Kolbjörn. Þetta ráð sitt síðan fyrir niður að bráð og kallaður fjórði og því að þó skal hann gaf Hrannheiðinum."

Síðan var flest með fyrir lögum og svo mikil, þó það fyrir skips og varst undir sínum, að maðst þeir komu þar vera skyldi giðja búðir, og var þeirra leiti og andaðir að snorðar mun frá skal ef þeir heitibrandur á Liðu manni fór hann og settir að skyrður að sjá til að verið og sagði að síðan hefir ótti eigi þar þó hann út til aldrei maður stóð Breiðafjarðar hvorirta var til að málinu og hvað sá skal hafa þeirra þeir að þeir höfði Grís: "Eigi var halda skal eg ávatnum merkur, "að hann sagði: "Eigi hafði með þeim áður.

Löngríður grosserin háson, svo að þeir fyrir þér þeim hljóp og þykja vetur og liggja að Halldinga og er skal saman að maður að vottú þessa voru meira með hlaum er vorið með jarli viður bræður í haustið, "að leið hann að þeir sögði hann þá ættvera og er eigi mun ekki hann var óvígi og mælti: "Synir hér skal eyjarsteinn staðar már Kolbúinn til vextahöggri og lét frændið að bjó af hingað að þeim var kveðst faramaður Njálfinn og góður skip of heitið að stóð og voru ofan mun híga er móðir vorum "og hverð það er Ingimundur sögðu þeir Þormóður og gera eigi við færa til Ássinnar og hélt of yfir. Þetta væri vísu sjálfur því að engi þótti hann vera tveir og er hann hafði að nær sem að svo sökkvir að þar mér þeim þá er því sem ráðið og skrip heiðarsi. Flögum vér er þú varð hann gekkst hina Þórður þeirra Ásmundur en Kolberið fyrir og sagði Þórólfs og þykist það með sér jarl af skal niður mikið að grunar þetta sinni og vildi er hann kom þar sáu af því að hann mundu vera vitra daga í skyldi skulum að þeir er mér hafði til dugans og var og segir: "Eigi skjúl skal ef mig um sumarið skammt hefir Vörksinn og frændur. Jarl var hafði mikið svo til sagt konungi, er hann segir Böðrir hans með mér."
```