package toygraph

// An allocator is a utility class used to manipulate the buffers for operations to act on.
// Each node has a unique allocator for every buffer that it creates.
// A value allocator can contain data about the buffer to allocate, for example how long the vectors that it allocates should be.
type Allocator[T any] interface {
	// Create a new pointer to the data type
	New() *T
	// Set the data type to its zero value
	Zero(*T)
	// Check if the provided pointer to a data type is valid for this allocator.
	// For example, if we are allocating vecotrs, we will check if the provided vector has the same length as the vectors the allocator creates.
	Allowed(*T) error
}
