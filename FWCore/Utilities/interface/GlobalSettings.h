#ifndef FWCore_Utilities_GlobalSettings
#define FWCore_Utilities_GlobalSettings

namespace cms {
  //forward declaration of editor class
  class GlobalSettingsEditor;

  class GlobalSettings final {
  public:
    //const accessors
    static const GlobalSettings& get();
    bool trace() const { return trace_; }

  private:
    friend class GlobalSettingsEditor;

    GlobalSettings();
    GlobalSettings(const GlobalSettings&) = delete;

    const GlobalSettings& operator=(const GlobalSettings&) = delete;

    //non-const accessors used by editor class
    static GlobalSettings& get_();
    void setTrace(bool t) { trace_ = t; }

    //members
    bool trace_;
  };

}

#endif
