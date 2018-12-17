/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_QTHREADSEXEC_HPP
#define KOKKOS_QTHREADSEXEC_HPP

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_QTHREADS )

#include <unistd.h>
#include <impl/Kokkos_Spinwait.hpp>
#include <qthread/barrier.h>

//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

class QthreadsExec;

typedef void (*QthreadsExecFunctionPointer)( QthreadsExec &, const void * );

class QthreadsExec {
public:

  /** \brief States of a worker thread */
  enum { Terminating ///<  Termination in progress
         , Inactive    ///<  Exists, waiting for work
         , Active      ///<  Exists, performing work
         , Rendezvous  ///<  Exists, waiting in a barrier or reduce

         , ScanCompleted
         , ScanAvailable
         , ReductionAvailable
  };
private:

  friend class Kokkos::Qthreads;

  QthreadsExec * const * m_pool_base ; ///< Base for pool fan-in
  const QthreadsExec * const * m_worker_base;
  const QthreadsExec * const * m_shepherd_base;

  void  * m_scratch ;  ///< Scratch memory [ reduce, team, shared ]
  int     m_reduce_end ;     ///< End of scratch reduction memory
  int     m_scratch_reduce_end ;
  int     m_scratch_thread_end ;
  int     m_shepherd_rank ;
  int     m_shepherd_size ;

  int     m_shepherd_worker_rank ;
  int     m_shepherd_worker_size ;

  int           m_numa_rank ;
  int           m_numa_core_rank ;
  int           m_pool_rank ;
  int           m_pool_rank_rev ;
  int           m_pool_size ;
  int           m_pool_fan_size ;
  int volatile  m_pool_state ;  ///< State for global synchronizations
  qt_barrier_t * m_barrier;


  /*
   *  m_worker_rank = m_shepherd_rank * m_shepherd_worker_size + m_shepherd_worker_rank
   *  m_worker_size = m_shepherd_size * m_shepherd_worker_size
   */
  int     m_worker_rank;
  int     m_worker_size;

  int mutable volatile m_worker_state;

  // Members for dynamic scheduling

  // This thread's owned work_range
  Kokkos::pair<long,long> m_work_range __attribute__((aligned(16))) ;
  // Team Offset if one thread determines work_range for others
  long m_team_work_index;



  static void global_lock();
  static void global_unlock();
  static bool spawn();

  static void execute_resize_scratch( QthreadsExec & , const void * );
  static void execute_sleep(          QthreadsExec & , const void * );

  QthreadsExec( const QthreadsExec & );
  QthreadsExec & operator = ( const QthreadsExec & );

  static void execute_serial( void (*)( QthreadsExec & , const void * ) );

public:

  KOKKOS_INLINE_FUNCTION int pool_rank() const { return m_pool_rank; }
  KOKKOS_INLINE_FUNCTION int pool_size() const { return m_pool_size; }
  KOKKOS_INLINE_FUNCTION int team_rank() const { return m_numa_rank; }
  KOKKOS_INLINE_FUNCTION int team_core_rank() const { return m_numa_core_rank; }
  inline long team_work_index() const { return m_team_work_index ; }

  static int get_thread_count();
  static QthreadsExec * get_thread( const int init_thread_rank );


  inline void * reduce_memory() const { return m_scratch ; }
  KOKKOS_INLINE_FUNCTION  void * scratch_memory() const
  { return reinterpret_cast<unsigned char *>(m_scratch) + m_scratch_reduce_end ; }

  KOKKOS_INLINE_FUNCTION  int volatile & state() { return m_pool_state ; }
  KOKKOS_INLINE_FUNCTION  QthreadsExec * const * pool_base() const { return m_pool_base ; }

  static void driver(void);

  ~QthreadsExec();
  QthreadsExec();


  static void * resize_scratch( size_t reduce_size , size_t thread_size );

  static void * root_reduce_scratch();

  static bool is_process();

  static void verify_is_process( const std::string & , const bool initialized );

  static int is_initialized();

  static void finalize();

  /* Given a requested team size, return valid team size */
  static unsigned team_size_valid( unsigned );

  static void print_configuration( std::ostream & , const bool detail = false );

  //------------------------------------

  static void wait_yield( volatile int & , const int );

  //------------------------------------
  // All-thread functions:

  /** Execute the input function on all available Qthreads workers. */
  static void exec_all( Qthreads &, QthreadsExecFunctionPointer, const void * );

  /** Barrier across all workers participating in the 'exec_all'. */
  void exec_all_barrier() const
  {
    const int rev_rank = m_worker_size - ( m_worker_rank + 1 );

    int n, j;
    //printf("execall barriering\n");
    // this should be a qthreads barrier. 
    //for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < m_worker_size ); n <<= 1 ) {
	  qt_barrier_enter(m_barrier);
    //}
    if ( rev_rank ) {
      m_worker_state = QthreadsExec::Inactive;
	  qt_barrier_enter(m_barrier);
      //Impl::spinwait_while_equal<int>( m_worker_state, QthreadsExec::Inactive );
    }

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < m_worker_size ); n <<= 1 ) {
      m_worker_base[j]->m_worker_state = QthreadsExec::Active;
    }
  }
  /** Barrier across workers within the shepherd with rank < team_rank. */
  void shepherd_barrier( const int team_size ) const
  {

    std::cout << "shepherd barriering\n";
    if ( m_shepherd_worker_rank < team_size ) {

      const int rev_rank = team_size - ( m_shepherd_worker_rank + 1 );

      int n, j;

      for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < team_size ); n <<= 1 ) {
        Impl::spinwait_while_equal<int>( m_shepherd_base[j]->m_worker_state, QthreadsExec::Active );
      }

      if ( rev_rank ) {
        m_worker_state = QthreadsExec::Inactive;
        Impl::spinwait_while_equal<int>( m_worker_state, QthreadsExec::Inactive );
      }

      for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < team_size ); n <<= 1 ) {
        m_shepherd_base[j]->m_worker_state = QthreadsExec::Active;
      }
    }
  }

  inline
  void fan_in() const
    {
 		qt_barrier_enter(m_barrier);
    }
  static int  in_parallel();

  static void fence();
  //static bool sleep();
  static bool wake();
  static void initialize (
    unsigned thread_count ,
    unsigned use_numa_count ,
    unsigned use_cores_per_numa ,
    bool allow_asynchronous_threadpool );


  /** Reduce across all workers participating in the 'exec_all'. */
  template< class FunctorType, class ReducerType, class ArgTag >
  inline
  void exec_all_reduce( const FunctorType & func, const ReducerType & reduce ) const
  {
    typedef Kokkos::Impl::if_c< std::is_same<InvalidType, ReducerType>::value, FunctorType, ReducerType > ReducerConditional;
    typedef typename ReducerConditional::type ReducerTypeFwd;
    typedef Kokkos::Impl::FunctorValueJoin< ReducerTypeFwd, ArgTag > ValueJoin;

    std::cout << "exec all reducing\n";
    const int rev_rank = m_worker_size - ( m_worker_rank + 1 );

    int n, j;

    //std::cout << "*m_scratch " << *m_scratch_alloc << std::endl; //<< " *m_worker_base[1].m_scratch_alloc " << *m_worker_base[1]->m_scratch_alloc << std::endl;
    std::cout << "rev_rank " << rev_rank << " m_worker_size " << m_worker_size << std::endl;
    //std::cout << "rev_rank " << rev_rank << " n " << 1 << " rev_rank & n " << rev_rank & 1 << " m_worker_size " << m_worker_size << std::endl;
    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < m_worker_size ); n <<= 1 ) {
      const QthreadsExec & fan = *m_worker_base[j];

      qt_barrier_enter(m_barrier);
      //Impl::spinwait_while_equal<int>( fan.m_worker_state, QthreadsExec::Active );

      ValueJoin::join( ReducerConditional::select( func, reduce ), m_scratch, fan.m_scratch );
      std::cout << "m_scratch " << m_scratch << " fan.m_scratch_alloc " << fan.m_scratch << std::endl;
    }

    std::cout << "rev_rank 1 " << rev_rank << " m_worker_size " << m_worker_size << std::endl;
    if ( rev_rank ) {
      //printf("m_worker_state %d\n", m_worker_state);
      m_worker_state = QthreadsExec::Inactive;
      //printf("spinwaiting\n");
      qt_barrier_enter(m_barrier);
      //printf("spinwaited\n");
    }

    std::cout << "for rev_rank 1 " << rev_rank << " m_worker_size " << m_worker_size << std::endl;
    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < m_worker_size ); n <<= 1 ) {
      //printf("m_worker_base %p n %d\n", m_worker_base, n);
      //m_worker_base[j]->m_worker_state = QthreadsExec::Active;
    }
    std::cout << "done\n";
  }

  /** Scan across all workers participating in the 'exec_all'. */
  template< class FunctorType, class ArgTag >
  inline
  void exec_all_scan( const FunctorType & func ) const
  {
    typedef Kokkos::Impl::FunctorValueInit< FunctorType, ArgTag > ValueInit;
    typedef Kokkos::Impl::FunctorValueJoin< FunctorType, ArgTag > ValueJoin;
    typedef Kokkos::Impl::FunctorValueOps<  FunctorType, ArgTag > ValueOps;

    std::cout << "exec all scanning\n";
    const int rev_rank = m_worker_size - ( m_worker_rank + 1 );

    int n, j;

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < m_worker_size ); n <<= 1 ) {
      Impl::spinwait_while_equal<int>( m_worker_base[j]->m_worker_state, QthreadsExec::Active );
    }

    if ( rev_rank ) {
      m_worker_state = QthreadsExec::Inactive;
      Impl::spinwait_while_equal<int>( m_worker_state, QthreadsExec::Inactive );
    }
    else {
      // Root thread scans across values before releasing threads.
      // Worker data is in reverse order, so m_worker_base[0] is the
      // highest ranking thread.

      // Copy from lower ranking to higher ranking worker.
      for ( int i = 1; i < m_worker_size; ++i ) {
        ValueOps::copy( func
                      , m_worker_base[i-1]->m_scratch
                      , m_worker_base[i]->m_scratch
                      );
      }

      ValueInit::init( func, m_worker_base[m_worker_size-1]->m_scratch );

      // Join from lower ranking to higher ranking worker.
      // Value at m_worker_base[n-1] is zero so skip adding it to m_worker_base[n-2].
      for ( int i = m_worker_size - 1; --i > 0; ) {
        ValueJoin::join( func, m_worker_base[i-1]->m_scratch, m_worker_base[i]->m_scratch );
      }
    }

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < m_worker_size ); n <<= 1 ) {
      m_worker_base[j]->m_worker_state = QthreadsExec::Active;
    }
  }

  //----------------------------------------

  template< class Type >
  inline
  volatile Type * shepherd_team_scratch_value() const
  { return (volatile Type*)( ( (unsigned char *) m_scratch ) + m_reduce_end ); }

  template< class Type >
  inline
  void shepherd_broadcast( Type & value, const int team_size, const int team_rank ) const
  {

    std::cout << "shepherd broadcasting\n";
    if ( m_shepherd_base ) {
      Type * const shared_value = m_shepherd_base[0]->shepherd_team_scratch_value<Type>();
      if ( m_shepherd_worker_rank == team_rank ) { *shared_value = value; }
      memory_fence();
      shepherd_barrier( team_size );
      value = *shared_value;
    }
  }

  template< class Type >
  inline
  Type shepherd_reduce( const int team_size, const Type & value ) const
  {
    volatile Type * const shared_value = shepherd_team_scratch_value<Type>();
    *shared_value = value;
    *shepherd_team_scratch_value<Type>() = value;

    std::cout << "shepherd reducing 1\n";
    memory_fence();

    const int rev_rank = team_size - ( m_shepherd_worker_rank + 1 );

    int n, j;

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < team_size ); n <<= 1 ) {
      Impl::spinwait_while_equal<int>( m_shepherd_base[j]->m_worker_state, QthreadsExec::Active );
    }

    if ( rev_rank ) {
      m_worker_state = QthreadsExec::Inactive;
      Impl::spinwait_while_equal<int>( m_worker_state, QthreadsExec::Inactive );
    }
    else {
      Type & accum = *m_shepherd_base[0]->shepherd_team_scratch_value<Type>();
      for ( int i = 1; i < n; ++i ) {
        accum += *m_shepherd_base[i]->shepherd_team_scratch_value<Type>();
      }
      for ( int i = 1; i < n; ++i ) {
        *m_shepherd_base[i]->shepherd_team_scratch_value<Type>() = accum;
      }

      memory_fence();
    }

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < team_size ); n <<= 1 ) {
      m_shepherd_base[j]->m_worker_state = QthreadsExec::Active;
    }

    return *shepherd_team_scratch_value<Type>();
  }

  template< class JoinOp >
  inline
  typename JoinOp::value_type
  shepherd_reduce( const int team_size
                 , const typename JoinOp::value_type & value
                 , const JoinOp & op ) const
  {
    typedef typename JoinOp::value_type Type;

    std::cout << "shepherd reducing 2\n";
    volatile Type * const shared_value = shepherd_team_scratch_value<Type>();
    *shared_value = value;
    *shepherd_team_scratch_value<Type>() = value;

    memory_fence();

    const int rev_rank = team_size - ( m_shepherd_worker_rank + 1 );

    int n, j;

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < team_size ); n <<= 1 ) {
      Impl::spinwait_while_equal<int>( m_shepherd_base[j]->m_worker_state, QthreadsExec::Active );
    }

    if ( rev_rank ) {
      m_worker_state = QthreadsExec::Inactive;
      Impl::spinwait_while_equal<int>( m_worker_state, QthreadsExec::Inactive );
    }
    else {
      volatile Type & accum = *m_shepherd_base[0]->shepherd_team_scratch_value<Type>();
      for ( int i = 1; i < team_size; ++i ) {
        op.join( accum, *m_shepherd_base[i]->shepherd_team_scratch_value<Type>() );
      }
      for ( int i = 1; i < team_size; ++i ) {
        *m_shepherd_base[i]->shepherd_team_scratch_value<Type>() = accum;
      }

      memory_fence();
    }

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < team_size ); n <<= 1 ) {
      m_shepherd_base[j]->m_worker_state = QthreadsExec::Active;
    }

    return *shepherd_team_scratch_value<Type>();
  }

  template< class Type >
  inline
  Type shepherd_scan( const int team_size
                    , const Type & value
                    ,       Type * const global_value = 0 ) const
  {
    *shepherd_team_scratch_value<Type>() = value;

    memory_fence();

    std::cout << "shepherd scan 1\n";
    const int rev_rank = team_size - ( m_shepherd_worker_rank + 1 );

    int n, j;

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < team_size ); n <<= 1 ) {
      Impl::spinwait_while_equal<int>( m_shepherd_base[j]->m_worker_state, QthreadsExec::Active );
    }

    if ( rev_rank ) {
      m_worker_state = QthreadsExec::Inactive;
      Impl::spinwait_while_equal<int>( m_worker_state, QthreadsExec::Inactive );
    }
    else {
      // Root thread scans across values before releasing threads.
      // Worker data is in reverse order, so m_shepherd_base[0] is the
      // highest ranking thread.

      // Copy from lower ranking to higher ranking worker.

      Type accum = *m_shepherd_base[0]->shepherd_team_scratch_value<Type>();
      for ( int i = 1; i < team_size; ++i ) {
        const Type tmp = *m_shepherd_base[i]->shepherd_team_scratch_value<Type>();
        accum += tmp;
        *m_shepherd_base[i-1]->shepherd_team_scratch_value<Type>() = tmp;
      }

      *m_shepherd_base[team_size-1]->shepherd_team_scratch_value<Type>() =
        global_value ? atomic_fetch_add( global_value, accum ) : 0;

      // Join from lower ranking to higher ranking worker.
      for ( int i = team_size; --i; ) {
        *m_shepherd_base[i-1]->shepherd_team_scratch_value<Type>() += *m_shepherd_base[i]->shepherd_team_scratch_value<Type>();
      }

      memory_fence();
    }

    for ( n = 1; ( ! ( rev_rank & n ) ) && ( ( j = rev_rank + n ) < team_size ); n <<= 1 ) {
      m_shepherd_base[j]->m_worker_state = QthreadsExec::Active;
    }

    return *shepherd_team_scratch_value<Type>();
  }

  //----------------------------------------

  static inline
  int align_alloc( int size )
  {
    enum { ALLOC_GRAIN = 1 << 6 /* power of two, 64bytes */ };
    enum { ALLOC_GRAIN_MASK = ALLOC_GRAIN - 1 };
    return ( size + ALLOC_GRAIN_MASK ) & ~ALLOC_GRAIN_MASK;
  }

  void shared_reset( Qthreads::scratch_memory_space & );

  void * exec_all_reduce_value() const {
    std::cout << "exec all reduce value 1\n";
    return m_scratch; }

  /* Dynamic Scheduling related functionality */
  // Initialize the work range for this thread
  inline void set_work_range(const long& begin, const long& end, const long& chunk_size) {
    m_work_range.first = (begin+chunk_size-1)/chunk_size;
    m_work_range.second = end>0?(end+chunk_size-1)/chunk_size:m_work_range.first;
    //printf("%s: begin %ld end %ld chunk_size %ld m_work_range.first %ld m_work_range.second %ld\n", __func__, begin, end, chunk_size, m_work_range.first, m_work_range.second);
  }

  // Claim and index from this thread's range from the beginning
  inline long get_work_index_begin () {
    Kokkos::pair<long,long> work_range_new = m_work_range;
    Kokkos::pair<long,long> work_range_old = work_range_new;
    if(work_range_old.first>=work_range_old.second)
      return -1;

    work_range_new.first+=1;

    bool success = false;
    while(!success) {
      work_range_new = Kokkos::atomic_compare_exchange(&m_work_range,work_range_old,work_range_new);
      success = ( (work_range_new == work_range_old) ||
                  (work_range_new.first>=work_range_new.second));
      work_range_old = work_range_new;
      work_range_new.first+=1;
    }
    if(work_range_old.first<work_range_old.second)
      return work_range_old.first;
    else
      return -1;
  }

  // Claim and index from this thread's range from the end
  inline long get_work_index_end () {
    Kokkos::pair<long,long> work_range_new = m_work_range;
    Kokkos::pair<long,long> work_range_old = work_range_new;
    if(work_range_old.first>=work_range_old.second)
      return -1;
    work_range_new.second-=1;
    bool success = false;
    while(!success) {
      work_range_new = Kokkos::atomic_compare_exchange(&m_work_range,work_range_old,work_range_new);
      success = ( (work_range_new == work_range_old) ||
                  (work_range_new.first>=work_range_new.second) );
      work_range_old = work_range_new;
      work_range_new.second-=1;
    }
    if(work_range_old.first<work_range_old.second)
      return work_range_old.second-1;
    else
      return -1;
  }

  long get_work_index(int team_size = 0) {
    long work_index = -1;
    work_index = get_work_index_begin();

    m_team_work_index = work_index;
    memory_fence();
    return work_index;
  }


  static void * exec_all_reduce_result();

  static void resize_worker_scratch( const int reduce_size, const int shared_size );
  static void clear_workers();

  //----------------------------------------

  inline int worker_rank() const { return m_worker_rank; }
  inline int worker_size() const { return m_worker_size; }
  inline int shepherd_worker_rank() const { return m_shepherd_worker_rank; }
  inline int shepherd_worker_size() const { return m_shepherd_worker_size; }
  inline int shepherd_rank() const { return m_shepherd_rank; }
  inline int shepherd_size() const { return m_shepherd_size; }

  static int worker_per_shepherd();
};

} // namespace Impl

} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

inline int Qthreads::in_parallel()
{ return Impl::QthreadsExec::in_parallel(); }

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
inline int Qthreads::is_initialized()
{ return Impl::QthreadsExec::is_initialized(); }
#else
inline int Qthreads::impl_is_initialized()
{ return Impl::QthreadsExec::is_initialized(); }
#endif

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
inline void Qthreads::initialize(
#else
inline void Qthreads::impl_initialize(
#endif
  unsigned threads_count ,
  unsigned use_numa_count ,
  unsigned use_cores_per_numa ,
  bool allow_asynchronous_threadpool )
{
  Impl::QthreadsExec::initialize( threads_count , use_numa_count , use_cores_per_numa , allow_asynchronous_threadpool );
}

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
inline void Qthreads::finalize()
#else
inline void Qthreads::impl_finalize()
#endif
{
  Impl::QthreadsExec::finalize();
}

inline void Qthreads::print_configuration( std::ostream & s , const bool detail )
{
  Impl::QthreadsExec::print_configuration( s , detail );
}

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
inline bool Qthreads::sleep()
{ /* return Impl::QthreadsExec::sleep() ;  */ return true; }

inline bool Qthreads::wake()
{ /* return Impl::QthreadsExec::wake() ; */ return true; }
#endif

inline void Qthreads::fence()
{ Impl::QthreadsExec::fence() ; }

} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif
#endif /* #define KOKKOS_QTHREADSEXEC_HPP */
