/* for std::this_thread */
#include <thread>

/* include C++ DDS API. */
#include "dds/dds.hpp"
// #include "dds/domain.hpp"
#include "dds/domain/DomainParticipant.hpp"

/* include the c++ data type, generated from idlcxx */
// #include "DataType.hpp"
#include <cstdio>
// class ExampleListener :
//                public virtual dds::domain::DomainParticipantListener
// {
// public:
//     virtual void on_inconsistent_topic (
//         dds::topic::AnyTopic& topic,
//         const dds::core::status::InconsistentTopicStatus& status)
//     {
//         std::cout << "on_inconsistent_topic" << std::endl;
//     }

//     virtual void on_offered_deadline_missed (
//         dds::pub::AnyDataWriter& writer,
//         const dds::core::status::OfferedDeadlineMissedStatus& status)
//     {
//         std::cout << "on_offered_deadline_missed" << std::endl;
//     }

//     virtual void on_offered_incompatible_qos (
//         dds::pub::AnyDataWriter& writer,
//         const dds::core::status::OfferedIncompatibleQosStatus& status)
//     {
//         std::cout << "on_offered_incompatible_qos" << std::endl;
//     }

//     virtual void on_liveliness_lost (
//         dds::pub::AnyDataWriter& writer,
//         const dds::core::status::LivelinessLostStatus& status)
//     {
//         std::cout << "on_liveliness_lost" << std::endl;
//     }

//     virtual void on_publication_matched (
//         dds::pub::AnyDataWriter& writer,
//         const dds::core::status::PublicationMatchedStatus& status)
//     {
//         std::cout << "on_publication_matched" << std::endl;
//     }

//     virtual void on_requested_deadline_missed (
//         dds::sub::AnyDataReader& reader,
//         const dds::core::status::RequestedDeadlineMissedStatus & status)
//     {
//         std::cout << "on_requested_deadline_missed" << std::endl;
//     }

//     virtual void on_requested_incompatible_qos (
//         dds::sub::AnyDataReader& reader,
//         const dds::core::status::RequestedIncompatibleQosStatus & status)
//     {
//         std::cout << "on_requested_incompatible_qos" << std::endl;
//     }

//     virtual void on_sample_rejected (
//         dds::sub::AnyDataReader& reader,
//         const dds::core::status::SampleRejectedStatus & status)
//     {
//         std::cout << "on_sample_rejected" << std::endl;
//     }

//     virtual void on_liveliness_changed (
//         dds::sub::AnyDataReader& reader,
//         const dds::core::status::LivelinessChangedStatus & status)
//     {
//         std::cout << "on_liveliness_changed" << std::endl;
//     }

//     virtual void on_data_available (
//         dds::sub::AnyDataReader& reader)
//     {
//         std::cout << "on_data_available" << std::endl;
//     }

//     virtual void on_subscription_matched (
//         dds::sub::AnyDataReader& reader,
//         const dds::core::status::SubscriptionMatchedStatus & status)
//     {
//         std::cout << "on_subscription_matched" << std::endl;
//     }

//     virtual void on_sample_lost (
//         dds::sub::AnyDataReader& reader,
//         const dds::core::status::SampleLostStatus & status)
//     {
//         std::cout << "on_sample_lost" << std::endl;
//     }

//     virtual void on_data_on_readers (
//         dds::sub::Subscriber& subs)
//     {
//         std::cout << "on_data_on_readers" << std::endl;
//     }
// };

int main() {
  dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id(),
                                           dds::domain::DomainParticipant::default_participant_qos(),
                                           new ExampleListener(),
                                           dds::core::status::StatusMask::all());
  std::printf("hiiiii");

  
  return 0;
}
