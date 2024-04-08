resource "azurerm_hdinsight_cluster" {
    name                = "mapper-reducer-hdicluster"
    resource_group_name = azurerm_resource_group.XXXXXXXX.name
    location            = azurerm_resource_group.XXXXXXXX.location
    cluster_version     = "4.0"
    tier                = "Standard"
    component_version {
      hadoop = "3.1"
    }
    gateway {
      enabled  = true
      username = "XXXXXXXXXXXXX"
      password = "XXXXXXXXXXXXX"
    }
    roles {
      head_node {
        vm_size  = "Standard_D3_V2"
        username = "XXXXXXXXXXXXXX"
        password = "XXXXXXXXXXXXXX"
      }
      worker_node {
        vm_size = "Standard_D3_V2"
        username = "XXXXXXXXXXXXXX"
        password = "XXXXXXXXXXXXXX"
        number_of_disks_per_node = 3
        target_instance_count    = 3
      }
      zookeeper_node {
        vm_size  = "Standard_D3_V2"
        username = "XXXXXXXXXXXXXX"
        password = "XXXXXXXXXXXXXX"
      }
    }
  }