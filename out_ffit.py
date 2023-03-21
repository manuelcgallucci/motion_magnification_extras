import matplotlib.pyplot as plt

fft_ = (289.74119515627456, 106.6373069912788, 259.13238079677546, 269.6212971473888, 640.0596973225099, 327.89639042504473, 378.94500307376745, 211.52429739488883, 163.29690023381724, 165.74790296901014, 190.426074333359, 129.2391521155264, 117.31667553067662, 42.417138061949395, 58.54750911597794, 59.38865767351747, 41.88960804597307, 26.19391764206774, 48.436913908334844, 22.110005116673253, 19.97239903046895, 3.892551289445971, 30.275244880213705, 12.47212577829546, 14.116047407694854, 15.923054346282596, 12.71637880313729, 13.511903383366725, 10.547066058496828, 6.756661590053061, 10.335447884851439, 7.415218264502372, 5.833625015189723, 25.758255716691338, 6.415624674441661, 2.044240893791839, 5.601155981130031, 18.79269039404716, 12.188160018783115, 6.705265248026753, 5.158843003056323, 
9.012083459210098, 7.90637561236392, 7.700286372096008, 5.097737434170026, 15.614450058212116, 12.813980465504882, 10.704610137461552, 6.881857597251292, 12.767145334803699, 6.232403234893142, 13.164683640429923, 3.3428701489850017, 6.217374684017854, 2.6605970087706927, 1.5657994776008906, 1.3458379988168108, 4.874847790919004, 6.3394318226337125, 2.9999999999999996, 10.012395813824705, 2.620262127748903, 8.933603834737882, 9.735383808219785, 4.9102645507364455, 6.5909366221018715, 10.047867114956407, 6.596750884071375, 4.765813976019216, 3.7953499088479314, 5.85908076819949, 3.382609001578074, 1.1503140372429348, 3.4663316277987017, 4.472135954999582, 1.62810301339755, 2.751592002226024, 5.421945022303745, 6.747145243454134, 5.663983528569871, 8.990360137943876, 5.563718964319427, 3.5100485223311138, 11.442985489892667, 6.289757115705563, 6.342184533009494, 3.3536423552971004, 10.744205231305816, 
7.875658029630104, 3.787812582150996, 0.9008890153292898, 3.5448238658436355, 6.784930056075696, 5.132187291994387, 3.0112848627467543, 4.464428690349352, 0.6336841238465415, 5.139920789501207, 7.733133651656939, 0.9999999999999951, 1.9584944621355176, 7.176544789403194, 2.4750015611532605, 4.831831792070533, 9.705573548478178, 4.665827921709177, 2.0277695347517763, 5.353009674665591, 2.889291773745725, 4.813216964833954, 2.766487223624168, 9.64552873282008, 7.583446821037278, 4.939622876050625, 6.485571530186583, 1.9855071623339278, 7.126470943754974, 8.0602043211566, 1.9164693034599956, 3.0000000000000004, 8.179172889840515, 2.187055029562927, 3.287536658609097, 7.595044992503983, 2.955876449331392, 2.9298534704767114, 3.1691709269396786, 3.176349672369137, 0.3835331315180248, 9.33064429149703, 2.4879800246982993, 4.000955233363954, 3.527748074433329, 1.858023381392474, 6.481056353853354, 1.7222390526966131, 2.93463497390861, 3.2615616716736673, 7.0706357101450825, 3.6202737855549647, 1.7262661000793014, 3.694329842098781, 7.039783685269005, 3.9928186204981304, 3.5415575499961216, 3.861267678415462, 4.2290204859723, 8.496393988789533, 9.67615178227111)


fft_no_flt = (198.3614366804764, 41.60777739925279, 92.37770345546735, 140.33986911244796, 364.0990889306562, 243.08787949145878, 244.7118456538458, 164.75601299724417, 101.35604554081108, 106.97768363808375, 104.35113682875641, 79.20488306609401, 55.36453235296034, 43.880939606822686, 63.3486499291793, 63.534062777848234, 39.58223038606026, 26.792870267151436, 28.90193681941898, 9.530353469343988, 29.901412886419674, 11.404012249250892, 25.513446834137564, 17.39716965685983, 17.428718169395506, 20.543990273616632, 21.680401475362036, 10.836783464669557, 14.341466346459777, 11.079521713672674, 23.805433950895036, 21.394599043552063, 20.021004226444827, 10.675543470167023, 21.03262635237047, 7.828926497683221, 7.235547881871962, 13.271580001349653, 18.375275233604857, 9.843539951552321, 17.090844163486675, 13.487284194307884, 5.820295280625001, 12.32513178220764, 15.326136874143453, 9.502169394464799, 3.0065257496650974, 17.368687786297627, 2.491753757502109, 4.5825756949558345, 14.000028955818289, 18.419209232473015, 14.547961560878605, 13.444621474321968, 8.346450613759623, 6.402277671215386, 9.676179551367687, 5.147596477129977, 7.62881461367717, 1.959833245139305, 4.165482010010606, 7.902970212352458, 6.367016062064377, 16.546284926317153, 6.282715447268307, 2.9856994577806635, 5.5945390978836205, 12.348560691732203, 5.101083530864335, 8.822665713560811, 8.734721550219335, 8.920469352492145, 4.865498996572264, 10.089855901901428, 7.211102550927973, 4.442348310107398, 
3.560623774011473, 8.56862186412953, 3.9409730607709923, 5.409485436999766, 4.20338727886333, 5.261274854582987, 8.584995724790447, 2.4106920414543085, 5.654657512334223, 5.646461203937309, 7.2200349562948976, 12.888066868427407, 8.224128867739289, 2.060145285230049, 8.464734326887639, 5.471694987741214, 7.056161224508788, 5.325156412604762, 1.9380234103984653, 5.012421897573043, 6.545821457336179, 4.745083313565555, 5.630057832749699, 5.567764362830023, 10.419606223717757, 1.9694212707452028, 5.2378028582166305, 0.9734972507339859, 5.107941646612768, 7.109569120160237, 5.944768543762671, 2.892533149250608, 7.255789704056632, 4.2088126830243455, 7.721071761441909, 6.413533203842584, 6.528596261209536, 3.916183638756408, 11.885870942726065, 15.108244099576776, 6.645974968652979, 8.303596048195073, 9.206146030007684, 8.55330659167826, 4.466394384994398, 7.940109878841034, 4.49588639283084, 6.728391156111079, 4.0298614085081415, 6.9972286360965255, 9.41441526194967, 7.507705344221574, 8.842483709454063, 3.0367200951097493, 6.657679710480626, 4.318468474752546, 9.76958079833616, 10.808443211201656, 9.693658380617219, 1.9006484118773157, 2.2137358071188604, 2.6922594126883053, 7.567444099885273, 3.4662012749469455, 8.957052047712354, 5.4397410951222875, 9.171013689736037, 1.8533968919537456, 7.303513056073442, 7.784915121332438, 5.5495512706580055, 6.215945266114912, 5.196534006760496)

fps = 30
xaxis=[i*(1/ fps) for i in range(len(fft_))]
            

plt.figure()
plt.plot(xaxis, fft_, 'r')
plt.plot(xaxis, fft_no_flt, 'b')
plt.title("fft")
plt.legend(["Using temporal filter","No filter"])
plt.savefig("./max_seq_fft_comparison.png")
plt.close()