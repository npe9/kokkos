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

#ifndef KOKKOS_QTHREADS_PARALLEL_HPP
#define KOKKOS_QTHREADS_PARALLEL_HPP

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_QTHREADS )

#include <vector>
#include <unistd.h>

#include <Kokkos_Parallel.hpp>

#include <impl/Kokkos_StaticAssert.hpp>
#include <impl/Kokkos_FunctorAdapter.hpp>

#include <KokkosExp_MDRangePolicy.hpp>

#include <Qthreads/Kokkos_QthreadsExec.hpp>

//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/* ParallelFor Kokkos::Qthreads with RangePolicy */

template< class FunctorType , class ... Traits >
class ParallelFor< FunctorType
                 , Kokkos::RangePolicy< Traits ... >
                 , Kokkos::Qthreads
                 >
{
private:

  typedef Kokkos::RangePolicy< Traits ... >  Policy ;
  typedef typename Policy::work_tag     WorkTag ;
  typedef typename Policy::member_type  Member ;
  typedef typename Policy::WorkRange    WorkRange ;

  const FunctorType  m_functor ;
  const Policy       m_policy ;

  template< class TagType >
  inline static
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_range( const FunctorType & functor ,
              //              const Member ibeg , const Member iend )
    const Member ibeg , const Member iend )
    {
#if defined( KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION ) &&  \
  defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
	 // printf("ibeg %llu iend %llu\n", (unsigned long long)ibeg, (unsigned long long)iend);
      for ( Member i = ibeg ; i < iend ; ++i ) {
        functor( i );
      }
    }

  template< class TagType >
  inline static
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_range( const FunctorType & functor ,
              const Member ibeg , const Member iend )
    {
      const TagType t{} ;
#if defined( KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION ) &&  \
  defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
	 // printf("ibeg %lu iend %lu\n", ibeg, iend);
      for ( Member i = ibeg ; i < iend ; ++i ) {
        functor( t , i );
      }
    }

  static void exec( QthreadsExec & exec , const void * arg )
  {
    exec_schedule<typename Policy::schedule_type::type>(exec,arg);
  }

  template<class Schedule>
  static
  typename std::enable_if< std::is_same<Schedule,Kokkos::Static>::value >::type
  exec_schedule( QthreadsExec & exec , const void * arg )
  {
    const ParallelFor & self = * ((const ParallelFor *) arg );

    WorkRange range( self.m_policy , exec.worker_rank() , exec.worker_size() );

    ParallelFor::template exec_range< WorkTag >
      ( self.m_functor , range.begin() , range.end() );

    exec.fan_in();
  }

  template<class Schedule>
  static
  typename std::enable_if< std::is_same<Schedule,Kokkos::Dynamic>::value >::type
  exec_schedule( QthreadsExec & exec , const void * arg )
  {
    const ParallelFor & self = * ((const ParallelFor *) arg );

    WorkRange range( self.m_policy , exec.worker_rank() , exec.worker_size() );

    exec.set_work_range(range.begin(),range.end(),self.m_policy.chunk_size());
    exec.exec_all_barrier();

    long work_index = exec.get_work_index();

    while(work_index != -1) {
      const Member begin = static_cast<Member>(work_index) * self.m_policy.chunk_size();
      const Member end = begin + self.m_policy.chunk_size() < self.m_policy.end()?begin+self.m_policy.chunk_size():self.m_policy.end();

      ParallelFor::template exec_range< WorkTag >
                                                              ( self.m_functor , begin , end );
      work_index = exec.get_work_index();
    }

    //exec.fan_in();
  }

public:

  inline
  void execute() const
    {
      QthreadsExec::exec_all(Qthreads::instance(), & ParallelFor::exec , this );
      //QthreadsExec::start( & ParallelFor::exec , this );
      //QthreadsExec::fence();
    }

  ParallelFor( const FunctorType & arg_functor
             , const Policy      & arg_policy
             )
    : m_functor( arg_functor )
    , m_policy(  arg_policy )
    {}
};



// MDRangePolicy impl
template< class FunctorType , class ... Traits >
class ParallelFor< FunctorType
                 , Kokkos::MDRangePolicy< Traits ... >
                 , Kokkos::Qthreads
                 >
{
private:
  typedef Kokkos::MDRangePolicy< Traits ... > MDRangePolicy ;
  typedef typename MDRangePolicy::impl_range_policy         Policy ;

  typedef typename MDRangePolicy::work_tag                  WorkTag ;

  typedef typename Policy::WorkRange   WorkRange ;
  typedef typename Policy::member_type Member ;

  typedef typename Kokkos::Impl::HostIterateTile< MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, void > iterate_type;

  const FunctorType   m_functor ;
  const MDRangePolicy m_mdr_policy ;
  const Policy        m_policy ;  // construct as RangePolicy( 0, num_tiles ).set_chunk_size(1) in ctor

  inline static
  void
  exec_range( const MDRangePolicy & mdr_policy 
            , const FunctorType & functor
            , const Member ibeg , const Member iend )
    {
      #if defined( KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION ) && \
          defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
      #pragma ivdep
      #endif
      for ( Member i = ibeg ; i < iend ; ++i ) {
        iterate_type( mdr_policy, functor )( i );
      }
    }

  static void exec( QthreadsExec & exec , const void * arg )
  {
   // printf("exec 1\n");
    exec_schedule<typename Policy::schedule_type::type>(exec,arg);
   // printf("execed 1\n");
  }

  template<class Schedule>
  static
  typename std::enable_if< std::is_same<Schedule,Kokkos::Static>::value >::type
  exec_schedule( QthreadsExec & exec , const void * arg )
  {
    const ParallelFor & self = * ((const ParallelFor *) arg );

    WorkRange range( self.m_policy , exec.worker_rank() , exec.worker_size() );
   // printf("%s: exec_range", __func__);
    ParallelFor::exec_range
      ( self.m_mdr_policy, self.m_functor , range.begin() , range.end() );
   // printf("%s: exec_ranged", __func__);
    //exec.fan_in();
  }

  template<class Schedule>
  static
  typename std::enable_if< std::is_same<Schedule,Kokkos::Dynamic>::value >::type
  exec_schedule( QthreadsExec & exec , const void * arg )
  {
    const ParallelFor & self = * ((const ParallelFor *) arg );

    WorkRange range( self.m_policy , exec.worker_rank() , exec.worker_size() );

    exec.set_work_range(range.begin(),range.end(),self.m_policy.chunk_size());
    //exec.reset_steal_target();
    exec.exec_all_barrier();

    long work_index = exec.get_work_index();

    while(work_index != -1) {
      const Member begin = static_cast<Member>(work_index) * self.m_policy.chunk_size();
      const Member end = begin + self.m_policy.chunk_size() < self.m_policy.end()?begin+self.m_policy.chunk_size():self.m_policy.end();

      ParallelFor::exec_range
        ( self.m_mdr_policy, self.m_functor , begin , end );
      work_index = exec.get_work_index();
    }

    //exec.fan_in();
  }

public:

  inline
  void execute() const
    {

      QthreadsExec::exec_all(Qthreads::instance(), & ParallelFor::exec , this );
      //QthreadsExec::start( & ParallelFor::exec , this );
      //QthreadsExec::fence();
    }

  ParallelFor( const FunctorType & arg_functor
             , const MDRangePolicy      & arg_policy )
    : m_functor( arg_functor )
    , m_mdr_policy( arg_policy )
    , m_policy( Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1) )
    {}
};


//----------------------------------------------------------------------------
/* ParallelFor Kokkos::Qthreads with TeamPolicy */

template< class FunctorType , class ... Properties >
class ParallelFor< FunctorType
                 , Kokkos::TeamPolicy< Properties ... >
                 , Kokkos::Qthreads
                 >
{
private:

  typedef Kokkos::Impl::TeamPolicyInternal< Kokkos::Qthreads, Properties ... >  Policy ;
  typedef typename Policy::work_tag                    WorkTag ;
  typedef typename Policy::member_type                 Member ;

  const FunctorType  m_functor ;
  const Policy       m_policy ;
  const int          m_shared ;

  template< class TagType , class Schedule>
  inline static
  typename std::enable_if< std::is_same< TagType , void >::value
  && std::is_same<Schedule,Kokkos::Static>::value >::type
  exec_team( const FunctorType & functor , Member member )
    {
//	printf("need to implement scheduling\n");
      //for ( ; member.valid_static() ; member.next_static() ) {
       // functor( member );
      //}
    }

  template< class TagType , class Schedule>
  inline static
  typename std::enable_if< ! std::is_same< TagType , void >::value
  && std::is_same<Schedule,Kokkos::Static>::value >::type
  exec_team( const FunctorType & functor , Member member )
    {
      const TagType t{} ;


//	printf("need to implement scheduling\n");
      //for ( ; member.valid_static() ; member.next_static() ) {
       // functor( t , member );
      //}
    }

  template< class TagType , class Schedule>
  inline static
  typename std::enable_if< std::is_same< TagType , void >::value
  && std::is_same<Schedule,Kokkos::Dynamic>::value >::type
  exec_team( const FunctorType & functor , Member member )
    {

//	printf("need to implement scheduling\n");
      //for ( ; member.valid_dynamic() ; member.next_dynamic() ) {
       // functor( member );
      //}
    }

  template< class TagType , class Schedule>
  inline static
  typename std::enable_if< ! std::is_same< TagType , void >::value
                          && std::is_same<Schedule,Kokkos::Dynamic>::value >::type
  exec_team( const FunctorType & functor , Member member )
    {
      const TagType t{} ;
      for ( ; member.valid_dynamic() ; member.next_dynamic() ) {
        functor( t , member );
      }
    }

  static void exec( QthreadsExec & exec , const void * arg )
  {
    const ParallelFor & self = * ((const ParallelFor *) arg );

    ParallelFor::exec_team< WorkTag , typename Policy::schedule_type::type >
      ( self.m_functor , Member( & exec , self.m_policy , self.m_shared ) );

    //exec.barrier();
    //exec.fan_in();
  }

public:

  inline
  void execute() const
    {
      QthreadsExec::resize_worker_scratch( 0 , Policy::member_type::team_reduce_size() + m_shared );

      QthreadsExec::exec_all(Qthreads::instance(), & ParallelFor::exec , this );
      //QthreadsExec::start( & ParallelFor::exec , this );

      //QthreadsExec::fence();
    }

  ParallelFor( const FunctorType & arg_functor
             , const Policy      & arg_policy )
    : m_functor( arg_functor )
    , m_policy(  arg_policy )
    //, m_shared( arg_policy.scratch_size(0) + arg_policy.scratch_size(1) + FunctorTeamShmemSize< FunctorType >::value( arg_functor , arg_policy.team_size() ) )
    , m_shared( 1 )
    { }
};


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/* ParallelReduce with Kokkos::Qthreads and RangePolicy */

template< class FunctorType , class ReducerType , class ... Traits >
class ParallelReduce< FunctorType
                    , Kokkos::RangePolicy< Traits ... >
                    , ReducerType
                    , Kokkos::Qthreads
                    >
{
private:

  typedef Kokkos::RangePolicy< Traits ... >  Policy ;

  typedef typename Policy::work_tag     WorkTag ;
  typedef typename Policy::WorkRange    WorkRange ;
  typedef typename Policy::member_type  Member ;

  typedef Kokkos::Impl::if_c< std::is_same<InvalidType, ReducerType>::value, FunctorType, ReducerType > ReducerConditional;
  typedef typename ReducerConditional::type ReducerTypeFwd;
  typedef typename Kokkos::Impl::if_c< std::is_same<InvalidType, ReducerType>::value, WorkTag, void >::type WorkTagFwd;

  typedef Kokkos::Impl::FunctorValueTraits< ReducerTypeFwd , WorkTagFwd > ValueTraits ;
  typedef Kokkos::Impl::FunctorValueInit<   ReducerTypeFwd , WorkTagFwd > ValueInit ;

  typedef typename ValueTraits::pointer_type    pointer_type ;
  typedef typename ValueTraits::reference_type  reference_type ;

  const FunctorType   m_functor ;
  const Policy        m_policy ;
  const ReducerType   m_reducer ;
  const pointer_type  m_result_ptr ;

  template< class TagType >
  inline static
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_range( const FunctorType & functor
            , const Member ibeg , const Member iend
            , reference_type update )
    {
#if defined( KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION ) &&  \
  defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
     // printf("exec_range 1\n");
      for ( Member i = ibeg ; i < iend ; ++i ) {
        functor( i , update );
      }

     // printf("exec_ranged 1\n");
    }

  template< class TagType >
  inline static
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_range( const FunctorType & functor
            , const Member ibeg , const Member iend
            , reference_type update )
    {
      const TagType t{} ;

     // printf("exec_range 2\n");
#if defined( KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION ) &&  \
  defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
#pragma ivdep
#endif
      for ( Member i = ibeg ; i < iend ; ++i ) {
        functor( t , i , update );
      }
     // printf("exec_ranged 2\n");
    }

  static void
  exec( QthreadsExec & exec , const void * arg ) {

   // printf("exec 2\n");
    exec_schedule<typename Policy::schedule_type::type>(exec, arg);
   // printf("execed 2\n");
  }

  template<class Schedule>
  static
  typename std::enable_if< std::is_same<Schedule,Kokkos::Static>::value >::type
  exec_schedule( QthreadsExec & exec , const void * arg )
  {
    const ParallelReduce & self = * ((const ParallelReduce *) arg );
    const WorkRange range( self.m_policy, exec.worker_rank(), exec.worker_size() );

   // printf("exec_schedule\n");
    ParallelReduce::template exec_range< WorkTag >
      ( self.m_functor , range.begin() , range.end()
      , ValueInit::init( ReducerConditional::select(self.m_functor , self.m_reducer) , exec.exec_all_reduce_value() ) );

    exec.template exec_all_reduce< FunctorType, ReducerType , WorkTag >( self.m_functor , self.m_reducer );
   // printf("exec_scheduled\n");
  }

  template<class Schedule>
  static
  typename std::enable_if< std::is_same<Schedule,Kokkos::Dynamic>::value >::type
    exec_schedule( QthreadsExec & exec , const void * arg )
  {
    const ParallelReduce & self = * ((const ParallelReduce *) arg );
    const WorkRange range( self.m_policy, exec.worker_rank(), exec.worker_size() );

    exec.set_work_range(range.begin(),range.end(),self.m_policy.chunk_size());
    //exec.reset_steal_target();
    exec.exec_all_barrier();

    long work_index = exec.get_work_index();
    reference_type update = ValueInit::init( ReducerConditional::select(self.m_functor , self.m_reducer) , exec.exec_all_reduce_value() );
    while(work_index != -1) {
      const Member begin = static_cast<Member>(work_index) * self.m_policy.chunk_size();
      const Member end = begin + self.m_policy.chunk_size() < self.m_policy.end()?begin+self.m_policy.chunk_size():self.m_policy.end();
      ParallelReduce::template exec_range< WorkTag >
        ( self.m_functor , begin , end
        , update );
      work_index = exec.get_work_index();
    }

    exec.template exec_all_reduce< FunctorType, ReducerType, WorkTag >( self.m_functor, self.m_reducer );
  }

public:

  inline
  void execute() const
    {
      QthreadsExec::resize_worker_scratch( ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) ) , 0 );

     // printf("execute parallel reduce\n");
      QthreadsExec::exec_all(Qthreads::instance(), & ParallelReduce::exec , this );
      //sleep(1);
      //QthreadsExec::start( & ParallelReduce::exec , this );

      //QthreadsExec::fence();
      std::cout << "m_result_ptr " << m_result_ptr << std::endl ;
      if ( m_result_ptr ) {

        const pointer_type data =
          (pointer_type) QthreadsExec::exec_all_reduce_result();

        const unsigned n = ValueTraits::value_count( ReducerConditional::select(m_functor , m_reducer) );
        for ( unsigned i = 0 ; i < n ; ++i ) {
        //  std::cout << "data " << data[i] << std::endl;
          m_result_ptr[i] = data[i];
        //  std::cout << "m_result_ptr " << m_result_ptr[i] << std::endl;
        }
      }
     // printf("executed parallel reduce\n");
    }

  template< class HostViewType >
  ParallelReduce( const FunctorType  & arg_functor ,
                  const Policy       & arg_policy ,
                  const HostViewType & arg_result_view ,
                  typename std::enable_if<Kokkos::is_view< HostViewType >::value &&
                                          !Kokkos::is_reducer_type< ReducerType >::value
                  , void*>::type = NULL)
    : m_functor( arg_functor )
    , m_policy( arg_policy )
    , m_reducer( InvalidType() )
    , m_result_ptr( arg_result_view.data() )
    {
     // printf("making parallel reduce with view\n");
      std::cout << "arg_result_view.data() " << arg_result_view.data() << std::endl;
      static_assert( Kokkos::is_view< HostViewType >::value
                     , "Kokkos::Qthreads reduce result must be a View" );

      static_assert( std::is_same< typename HostViewType::memory_space , HostSpace >::value
                     , "Kokkos::Qthreads reduce result must be a View in HostSpace" );
    }

  inline
  ParallelReduce( const FunctorType & arg_functor
                , Policy       arg_policy
                , const ReducerType& reducer )
    : m_functor( arg_functor )
    , m_policy( arg_policy )
    , m_reducer( reducer )
    , m_result_ptr( reducer.view().data() )
    {
     // printf("making parallel reduce with reducer\n");
      /*
      static_assert( std::is_same< typename ViewType::memory_space, HostSpace >::value
        , "Reduction result on Kokkos::Qthreads must be a Kokkos::View in HostSpace" );
      */
    }
};

//----------------------------------------------------------------------------
/* ParallelReduce with Kokkos::Qthreads and TeamPolicy */

template< class FunctorType , class ReducerType, class ... Properties >
class ParallelReduce< FunctorType
                    , Kokkos::TeamPolicy< Properties ... >
                    , ReducerType
                    , Kokkos::Qthreads
                    >
{
private:

  typedef Kokkos::Impl::TeamPolicyInternal< Kokkos::Qthreads, Properties ... >              Policy ;
  typedef typename Policy::work_tag                                WorkTag ;
  typedef typename Policy::member_type                             Member ;

  typedef Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, FunctorType, ReducerType> ReducerConditional;
  typedef typename ReducerConditional::type ReducerTypeFwd;
  typedef typename Kokkos::Impl::if_c< std::is_same<InvalidType,ReducerType>::value, WorkTag, void>::type WorkTagFwd;

  typedef Kokkos::Impl::FunctorValueTraits< ReducerTypeFwd , WorkTagFwd > ValueTraits ;
  typedef Kokkos::Impl::FunctorValueInit<   ReducerTypeFwd , WorkTagFwd > ValueInit ;

  typedef typename ValueTraits::pointer_type    pointer_type ;
  typedef typename ValueTraits::reference_type  reference_type ;

  const FunctorType  m_functor ;
  const Policy       m_policy ;
  const ReducerType  m_reducer ;
  const pointer_type m_result_ptr ;
  const int          m_shared ;

  template< class TagType >
  inline static
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_team( const FunctorType & functor , Member member , reference_type update )
    {
//	printf("need to enable\n");
      //for ( ; member.valid_static() ; member.next_static() ) {
       // functor( member , update );
      //}
    }

  template< class TagType >
  inline static
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_team( const FunctorType & functor , Member member , reference_type update )
    {
      const TagType t{} ;
//	printf("need to enable\n");
      //for ( ; member.valid_static() ; member.next_static() ) {
       // functor( t , member , update );
      //}
    }

  static void exec( QthreadsExec & exec , const void * arg )
  {
    const ParallelReduce & self = * ((const ParallelReduce *) arg );

    ParallelReduce::template exec_team< WorkTag >
      ( self.m_functor , Member( & exec , self.m_policy , self.m_shared )
      , ValueInit::init( ReducerConditional::select(self.m_functor , self.m_reducer) , exec.exec_all_reduce_value() ) );
   // printf("executing all reduce\n");
    exec.template exec_all_reduce< FunctorType, ReducerType, WorkTag >( self.m_functor, self.m_reducer );
  }

public:

  inline
  void execute() const
    {
     // printf("executing team reduce\n");
      QthreadsExec::resize_worker_scratch( ValueTraits::value_size( ReducerConditional::select(m_functor , m_reducer) ) , 0 );
      QthreadsExec::exec_all(Qthreads::instance(), & ParallelReduce::exec , this );
      //QthreadsExec::start( & ParallelReduce::exec , this );

      //QthreadsExec::fence();

      if ( m_result_ptr ) {

        const pointer_type data = (pointer_type) QthreadsExec::exec_all_reduce_result();

        const unsigned n = ValueTraits::value_count( ReducerConditional::select(m_functor , m_reducer) );
        for ( unsigned i = 0 ; i < n ; ++i ) { m_result_ptr[i] = data[i]; }
      }
    }

  template< class HostViewType >
  inline
  ParallelReduce( const FunctorType  & arg_functor ,
                  const Policy       & arg_policy ,
                  const HostViewType     & arg_result ,
                  typename std::enable_if<
                    Kokkos::is_view< HostViewType >::value &&
                    !Kokkos::is_reducer_type<ReducerType>::value
                    ,void*>::type = NULL)
    : m_functor( arg_functor )
    , m_policy(  arg_policy )
    , m_reducer( InvalidType() )
    , m_result_ptr( arg_result.data() )
    //, m_shared( arg_policy.scratch_size(0) + arg_policy.scratch_size(1) + FunctorTeamShmemSize< FunctorType >::value( arg_functor , arg_policy.team_size() ) )
    , m_shared( 1024*1024 )
    {
     // printf("making another parallel reduce with reducer\n");
    }

  inline
  ParallelReduce( const FunctorType & arg_functor
    , Policy       arg_policy
    , const ReducerType& reducer )
  : m_functor( arg_functor )
  , m_policy(  arg_policy )
  , m_reducer( reducer )
  , m_result_ptr(  reducer.view().data() )
  //, m_shared( arg_policy.scratch_size(0) + arg_policy.scratch_size(1) + FunctorTeamShmemSize< FunctorType >::value( arg_functor , arg_policy.team_size() ) )
  , m_shared( 1024*1024 )
  {
   // printf("making another parallel reduce with reducer\n");
  /*static_assert( std::is_same< typename ViewType::memory_space
                          , Kokkos::HostSpace >::value
  , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace" );*/
  }
};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/* ParallelScan with Kokkos::Qthreads and RangePolicy */

template< class FunctorType , class ... Traits >
class ParallelScan< FunctorType
                  , Kokkos::RangePolicy< Traits ... >
                  , Kokkos::Qthreads
                  >
{
private:

  typedef Kokkos::RangePolicy< Traits ... > Policy ;
  typedef typename Policy::WorkRange                               WorkRange ;
  typedef typename Policy::work_tag                                WorkTag ;
  typedef typename Policy::member_type                             Member ;
  typedef Kokkos::Impl::FunctorValueTraits< FunctorType, WorkTag > ValueTraits ;
  typedef Kokkos::Impl::FunctorValueInit<   FunctorType, WorkTag > ValueInit ;

  typedef typename ValueTraits::pointer_type    pointer_type ;
  typedef typename ValueTraits::reference_type  reference_type ;

  const FunctorType  m_functor ;
  const Policy       m_policy ;

  template< class TagType >
  inline static
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_range( const FunctorType & functor
            , const Member & ibeg , const Member & iend
            , reference_type update , const bool final )
    {
      #if defined( KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION ) && \
          defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
      #pragma ivdep
      #endif
     // printf("execing range parallel scan\n");
      for ( Member i = ibeg ; i < iend ; ++i ) {
        functor( i , update , final );
      }
    }

  template< class TagType >
  inline static
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_range( const FunctorType & functor
            , const Member & ibeg , const Member & iend
            , reference_type update , const bool final )
    {
      const TagType t{} ;
      #if defined( KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION ) && \
          defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
      #pragma ivdep
      #endif

     // printf("execing range 2 parallel scan\n");
      for ( Member i = ibeg ; i < iend ; ++i ) {
        functor( t , i , update , final );
      }
    }

  static void exec( QthreadsExec & exec , const void * arg )
  {
    const ParallelScan & self = * ((const ParallelScan *) arg );

    const WorkRange range( self.m_policy, exec.worker_rank(), exec.worker_size() );

   // printf("execing regular 2 parallel scan\n");
    reference_type update =
      ValueInit::init( self.m_functor , exec.exec_all_reduce_value() );

    ParallelScan::template exec_range< WorkTag >
      ( self.m_functor , range.begin(), range.end(), update, false );

    //  exec.template scan_large<FunctorType,WorkTag>( self.m_functor );
    // exec.template scan_small<FunctorType,WorkTag>( self.m_functor );

    ParallelScan::template exec_range< WorkTag >
      ( self.m_functor , range.begin(), range.end(), update, true );

    //exec.fan_in();
  }

public:

  inline
  void execute() const
    {

     // printf("execing inline parallel scan\n");
      QthreadsExec::resize_worker_scratch( 2 * ValueTraits::value_size( m_functor ) , 0 );
      QthreadsExec::exec_all(Qthreads::instance(), & ParallelScan::exec , this );
      // QthreadsExec::start( & ParallelScan::exec , this );
      // QthreadsExec::fence();
    }

  ParallelScan( const FunctorType & arg_functor
              , const Policy      & arg_policy )
    : m_functor( arg_functor )
    , m_policy( arg_policy )
  { 
	//printf("making scan\n"); 
  }
};


template< class FunctorType, class ReturnType, class ... Traits >
class ParallelScanWithTotal< FunctorType
                           , Kokkos::RangePolicy< Traits ... >
                           , ReturnType
                           , Kokkos::Qthreads
                           >
{
private:

  typedef Kokkos::RangePolicy< Traits ... > Policy ;
  typedef typename Policy::WorkRange                               WorkRange ;
  typedef typename Policy::work_tag                                WorkTag ;
  typedef typename Policy::member_type                             Member ;
  typedef Kokkos::Impl::FunctorValueTraits< FunctorType, WorkTag > ValueTraits ;
  typedef Kokkos::Impl::FunctorValueInit<   FunctorType, WorkTag > ValueInit ;

  typedef typename ValueTraits::pointer_type    pointer_type ;
  typedef typename ValueTraits::reference_type  reference_type ;

  const FunctorType  m_functor ;
  const Policy       m_policy ;
  ReturnType       & m_returnvalue;

  template< class TagType >
  inline static
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  exec_range( const FunctorType & functor
            , const Member & ibeg , const Member & iend
            , reference_type update , const bool final )
    {
      #if defined( KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION ) && \
          defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
      #pragma ivdep
      #endif
      for ( Member i = ibeg ; i < iend ; ++i ) {
        functor( i , update , final );
      }
    }

  template< class TagType >
  inline static
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  exec_range( const FunctorType & functor
            , const Member & ibeg , const Member & iend
            , reference_type update , const bool final )
    {
      const TagType t{} ;
      #if defined( KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION ) && \
          defined( KOKKOS_ENABLE_PRAGMA_IVDEP )
      #pragma ivdep
      #endif
      for ( Member i = ibeg ; i < iend ; ++i ) {
        functor( t , i , update , final );
      }
    }

  static void exec( QthreadsExec & exec , const void * arg )
  {
    const ParallelScanWithTotal & self = * ((const ParallelScanWithTotal *) arg );

    const WorkRange range( self.m_policy, exec.worker_rank(), exec.worker_size() );

    reference_type update =
      ValueInit::init( self.m_functor , exec.exec_all_reduce_value() );

   // printf("execing range for parallel scan\n");
    ParallelScanWithTotal::template exec_range< WorkTag >
      ( self.m_functor , range.begin(), range.end(), update, false );

    //  exec.template scan_large<FunctorType,WorkTag>( self.m_functor );
    // exec.template scan_small<FunctorType,WorkTag>( self.m_functor );

    ParallelScanWithTotal::template exec_range< WorkTag >
      ( self.m_functor , range.begin(), range.end(), update, true );

    //exec.fan_in();

    if (exec.worker_rank()==exec.worker_size()-1) {
      self.m_returnvalue = update;
    }
  }

public:

  inline
  void execute() const
    {

     // printf("executing parallel scan\n");
      QthreadsExec::resize_worker_scratch( 2 * ValueTraits::value_size( m_functor ) , 0 );
      //QthreadsExec::exec_all(Qthreads::instance(), & ParallelFor::exec , this );
      //QthreadsExec::start( & ParallelScanWithTotal::exec , this );
      //QthreadsExec::fence();
    }

  ParallelScanWithTotal( const FunctorType & arg_functor
                       , const Policy      & arg_policy
                       , ReturnType        & arg_returnvalue )
    : m_functor( arg_functor )
    , m_policy( arg_policy )
    , m_returnvalue(  arg_returnvalue )
    { }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif
#endif /* #define KOKKOS_QTHREADS_PARALLEL_HPP */

