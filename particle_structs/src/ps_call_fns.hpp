template <class DataTypes, typename MemSpace>
ParticleStructure<DataTypes, MemSpace>::~ParticleStructure() {
  if (scs)
    delete scs;
}
template <class DataTypes, typename MemSpace>
lid_t ParticleStructure<DataTypes, MemSpace>::nElems() const {
  if (scs)
    scs->nElems();
}
template <class DataTypes, typename MemSpace>
lid_t ParticleStructure<DataTypes, MemSpace>::nPtcls() const {
  if (scs)
    scs->nPtcls();

}
template <class DataTypes, typename MemSpace>
lid_t ParticleStructure<DataTypes, MemSpace>::capacity() const {
  if (scs)
    scs->capacity();

}
template <class DataTypes, typename MemSpace>
lid_t ParticleStructure<DataTypes, MemSpace>::nRows() const {
  if (scs)
    scs->nRows();
}

template <class DataTypes, typename MemSpace>
template <std::size_t N>
Segment<ParticleStructure<DataTypes, MemSpace>::DataType<N>, MemSpace>
ParticleStructure<DataTypes, MemSpace>::get() {
  if (scs)
    scs->get<N>();
}

template <class DataTypes, typename MemSpace>
void ParticleStructure<DataTypes, MemSpace>::rebuild(kkLidView new_element,
             kkLidView new_particle_elements = kkLidView(),
             MemberTypeViews<DataTypes> new_particle_info = NULL) {
  if (scs)
    scs->rebuild(new_element, new_particle_elements, new_particle_info);
}
template <class DataTypes, typename MemSpace>
void ParticleStructure<DataTypes, MemSpace>::migrate(kkLidView new_element,
             kkLidView new_process,
             kkLidView new_particle_elements = kkLidView(),
             MemberTypeViews<DataTypes> new_particle_info = NULL) {
  if (scs)
    scs->migrate(new_element, new_process);
}

template <class DataTypes, typename MemSpace>
template <typename FunctionType>
void ParticleStructure<DataTypes, MemSpace>::parallel_for(FunctionType& fn,
                                                          std::string name="") {
  if (scs)
    scs->parallel_for(fn, name);
}

template <class DataTypes, typename MemSpace>
void ParticleStructure<DataTypes, MemSpace>::printMetrics() const {
  if (scs)
    scs->printMetrics();
}
