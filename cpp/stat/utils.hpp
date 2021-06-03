#ifndef STAT_UTILS_HPP_
#define STAT_UTILS_HPP_

namespace sferes {
    namespace stat {

    std::string add_leading_zeros(const std::string& original_string, 
        size_t total_number_zeros = 7) {

      return std::string(total_number_zeros - original_string.length(), '0') + original_string;
    }

    std::string add_leading_zeros(const int original_number,
         size_t total_number_zeros = 7) {
         
      std::string original_string = boost::lexical_cast<std::string>(original_number);
      return std::string(total_number_zeros - original_string.length(), '0') + original_string;
    }
    }
}

#endif /* ifndef STAT_UTILS_HPP_



 */
