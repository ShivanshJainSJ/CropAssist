const statesAndCities = {
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Tirupati", "Nellore"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzzafarpur", "Purnia"],
    "Karnataka": ["Bengaluru", "Mysuru", "Hubballi", "Mangalore", "Belagavi"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Trichy", "Salem"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Agra", "Varanasi", "Meerut"],
    "West Bengal": ["Kolkata", "Howrah", "Siliguri", "Durgapur", "Asansol"],
    "Rajasthan": ["Jaipur", "Udaipur", "Jodhpur", "Kota", "Ajmer"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
    "Punjab": ["Chandigarh", "Amritsar", "Ludhiana", "Jalandhar", "Patiala"],
    "Haryana": ["Chandigarh", "Faridabad", "Gurugram", "Ambala", "Hisar"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain"],
    "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Kottayam", "Thrissur"],
    "Delhi": ["New Delhi", "Dwarka", "Vasant Kunj", "Connaught Place", "Saket"],
    "Uttarakhand": ["Dehradun", "Nainital", "Haridwar", "Rishikesh", "Roorkee"],
    "Himachal Pradesh": ["Shimla", "Manali", "Kullu", "Dharamshala", "Kangra"],
    "Chhattisgarh": ["Raipur", "Bilaspur", "Durg", "Korba", "Raigarh"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela", "Berhampur", "Sambalpur"],
    "Assam": ["Guwahati", "Dibrugarh", "Jorhat", "Silchar", "Tezpur"],
    "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad", "Hazaribagh", "Deoghar"],
    "Goa": ["Panaji", "Margao", "Vasco da Gama", "Mapusa", "Ponda"],
    "Telangana": ["Hyderabad", "Warangal", "Khammam", "Nizamabad", "Karimnagar"],
    "Andaman and Nicobar Islands": ["Port Blair", "Car Nicobar", "Mayabunder", "Diglipur", "Hut Bay"],
    "Lakshadweep": ["Kavaratti", "Agatti", "Amini", "Kadmat", "Kalapeni"],
    "Sikkim": ["Gangtok", "Mangan", "Namchi", "Jorethang", "Rangpo"],
    "Arunachal Pradesh": ["Itanagar", "Tawang", "Ziro", "Pasighat", "Bomdila"],
    "Nagaland": ["Kohima", "Dimapur", "Mokokchung", "Mon", "Tuensang"]
  };
  
  // Function to populate the state dropdown
  function print_state(id) {
    const statesDropdown = document.getElementById(id);
    Object.keys(statesAndCities).forEach(state => {
      const option = document.createElement("option");
      option.value = state;
      option.textContent = state;
      statesDropdown.appendChild(option);
    });
  }
  
  // Function to populate the city dropdown based on selected state
  function print_city(cityDropdownId) {
    const state = document.getElementById("sts").value; // Get the selected state
    const cities = statesAndCities[state] || []; // Get the cities for the selected state
    const cityDropdown = document.getElementById(cityDropdownId);
    cityDropdown.innerHTML = ""; // Clear previous options
    
    cities.forEach(city => {
      const option = document.createElement("option");
      option.value = city;
      option.textContent = city;
      cityDropdown.appendChild(option);
    });
  }
  