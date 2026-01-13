package org.llm4s.agent.memory

import com.dimafeng.testcontainers.PostgreSQLContainer
import org.scalatest.BeforeAndAfterAll
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.testcontainers.utility.DockerImageName
import org.testcontainers.DockerClientFactory

import java.util.UUID
import scala.util.Try

class PostgresMemoryStoreSpec extends AnyFlatSpec with Matchers with BeforeAndAfterAll with org.scalatest.Assertions {

  private lazy val container: PostgreSQLContainer =
    PostgreSQLContainer(
      dockerImageNameOverride = DockerImageName.parse("pgvector/pgvector:pg16")
    )

  override def beforeAll(): Unit = {
    val isDockerAvailable = Try(DockerClientFactory.instance().isDockerAvailable).getOrElse(false)
    assume(isDockerAvailable, "Docker is not available. Skipping integration tests.")
    super.beforeAll()
    container.start()
  }

  override def afterAll(): Unit = {
    val isDockerAvailable = Try(DockerClientFactory.instance().isDockerAvailable).getOrElse(false)
    if (isDockerAvailable) {
      container.stop()
    }
    super.afterAll()
  }

  private def createStore(table: String): PostgresMemoryStore = {
    val config = PostgresMemoryStore.Config(
      host = container.host,
      port = container.mappedPort(5432),
      database = container.databaseName,
      user = container.username,
      password = container.password,
      tableName = table,
      maxPoolSize = 4
    )

    PostgresMemoryStore(config).fold(
      err => fail(err.message),
      identity
    )
  }

  it should "store and retrieve a conversation memory" in {
    val store = createStore("agent_memories")
    val id    = MemoryId(UUID.randomUUID().toString)

    val memory = Memory(
      id = id,
      content = "Hello, I am a test memory",
      memoryType = MemoryType.Conversation,
      metadata = Map("conversation_id" -> "conv-1")
    )

    store.store(memory).isRight shouldBe true

    val retrieved = store.get(id).toOption.flatten
    retrieved shouldBe defined
    retrieved.get.content shouldBe "Hello, I am a test memory"
    retrieved.get.metadata("conversation_id") shouldBe "conv-1"

    store.close()
  }

  it should "persist data across store instances" in {
    val table = "persistent_memories"
    val id    = MemoryId(UUID.randomUUID().toString)

    val store1 = createStore(table)
    store1.store(Memory(id, "Persistence Check", MemoryType.Task)).isRight shouldBe true
    store1.close()

    val store2 = createStore(table)
    store2.get(id).toOption.flatten.map(_.content) shouldBe Some("Persistence Check")
    store2.close()
  }
}
