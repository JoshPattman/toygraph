package toygraph

import (
	"fmt"
	"math"
	"testing"
	"time"

	"gonum.org/v1/gonum/mat"
)

func TestNums(t *testing.T) {
	realEq := func(a, x, c float64) float64 {
		return math.Exp(a*x*x + c)
	}
	realDiff := func(a, x, c float64) (float64, float64, float64) {
		dydx := 2 * a * x * math.Exp(a*x*x+c)
		dyda := x * x * math.Exp(a*x*x+c)
		dydc := math.Exp(a*x*x + c)
		return dyda, dydx, dydc
	}

	g := NewGraph()
	aN := NumValue[float64](g)
	xN := NumValue[float64](g)
	cN := NumValue[float64](g)
	xxN := NumMul(g, xN.C, xN.C)
	axxN := NumMul(g, aN.C, xxN.C)
	axxPcN := NumAdd(g, axxN.C, cN.C)
	resultN := NumExp(g, axxPcN.C)

	a, x, c := 0.5, 0.7, -0.3

	startTime := time.Now()
	realEqResult := realEq(a, x, c)
	realTime := time.Since(startTime)
	startTime = time.Now()
	realDiffResultA, realDiffResultX, realDiffResultC := realDiff(a, x, c)
	realDiffTime := time.Since(startTime)
	startTime = time.Now()
	Set(aN.C, a)
	Set(xN.C, x)
	Set(cN.C, c)
	g.AllForward()
	graphEqResult := Get(resultN.C)
	graphTime := time.Since(startTime)
	if graphEqResult != realEqResult {
		t.Fatalf("result of graph (%.4f) was not the same as real equation (%.4f)", graphEqResult, realEqResult)
	}
	startTime = time.Now()
	g.AllZeroGrads()
	SetGrad(resultN.C, 1)
	g.AllBackward()
	graphDiffResultA, graphDiffResultX, graphDiffResultC := GetGrad(aN.C), GetGrad(xN.C), GetGrad(cN.C)
	graphGradTime := time.Since(startTime)

	if realDiffResultA != graphDiffResultA || realDiffResultX != graphDiffResultX {
		t.Fatalf("graph gradients (%.4f, %.4f, %.4f) did not match real gradients (%.4f, %.4f, %.4f)", graphDiffResultA, graphDiffResultX, graphDiffResultC, realDiffResultA, realDiffResultX, realDiffResultC)
	}
	fmt.Println("Graph time:", graphTime)
	fmt.Println("Graph grad time:", graphGradTime)
	fmt.Println("Real time:", realTime)
	fmt.Println("Real grad time:", realDiffTime)
}

func TestMatMul(t *testing.T) {
	g := NewGraph()

	inVec := VecValue(g, 3)
	inMat := MatValue(g, 2, 3)
	result := MatMulVec(g, inMat.C, inVec.C)

	Set(inVec.C, mat.NewVecDense(3, []float64{1, 2, 3}))
	Set(inMat.C, mat.NewDense(2, 3, []float64{
		1, 1, 1,
		0.5, 0, -0.5,
	}))

	g.AllForward()

	g.AllZeroGrads()
	SetGrad(result.C, mat.NewVecDense(2, []float64{1, 1}))
	g.AllBackward()

	targetResult := mat.NewVecDense(2, []float64{6, -1})
	targetGradVec := mat.NewVecDense(3, []float64{1.5, 1, 0.5})

	graphResult := Get(result.C)
	graphGradVec := GetGrad(inVec.C)

	for i := range graphResult.Len() {
		if graphResult.AtVec(i) != targetResult.AtVec(i) {
			t.Fatalf("results did not match %v and %v", graphResult, targetResult)
		}
	}

	for i := range graphResult.Len() {
		if graphGradVec.AtVec(i) != targetGradVec.AtVec(i) {
			t.Fatalf("results did not match %v and %v", graphGradVec, targetGradVec)
		}
	}

	// TODO: Check the gradients of the matrix - i actually don't know how to figure this out by hand though so ill come back to it
}
