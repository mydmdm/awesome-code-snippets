#ifndef __MATMUL_DEF_H__
#define __MATMUL_DEF_H__

#ifndef D
#define D float
#endif

#ifndef M
#define M 2000
#endif

#ifndef N
#define N 2000
#endif

#ifndef K
#define K 2000
#endif

#ifndef TILE
#define TILE 16
#endif

#ifndef num_repeat
#define num_repeat 100
#endif

enum class AlgoSel {
	all=0,
	cublas=1,
	naive=2,
	tiled=3
};

#endif
